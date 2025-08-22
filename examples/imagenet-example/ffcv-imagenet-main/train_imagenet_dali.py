import torch as ch
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
from tqdm import tqdm
import os
import time
import json
from uuid import uuid4
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

# 添加DALI相关导入
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as dalitorch
from nvidia.dali.plugin.pytorch import LastBatchPolicy

Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())), default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, 'path to training dataset', required=True),
    val_dataset=Param(str, 'path to validation dataset', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    batch_size=Param(int, 'The batch size', default=512),
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0)
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)
    
class ImageNetTrainer:
    @param('training.distributed')
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.initialize_logger()
        

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr
        }

        return lr_schedules[lr_schedule_type](epoch)
    
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing):
        assert optimizer == 'sgd'

        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]

        self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('data.batch_size')
    @param('training.distributed')
    def create_train_loader(self, train_dataset, num_workers, batch_size, distributed):
        # 定义DALI训练pipeline
        @pipeline_def
        
        def train_pipeline():
            images, labels = fn.readers.file(
                file_root=train_dataset,
                random_shuffle=True,
                pad_last_batch=True,
                name="Reader")
            
            # 解码图像
            images = fn.decoders.image(images, device="cpu", output_type=types.RGB)
            
            # 数据增强
            images = fn.random_resized_crop(
                images, 
                size=224,
                random_area=(0.08, 1.0))
            
            # 随机水平翻转
            images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
            

            # 归一化
            images = fn.crop_mirror_normalize(
                images,
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],  # 乘以255因为DALI期望输入在[0,255]范围
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],  # DALI使用stddev而不是std
                scale=1./255,
                dtype=types.FLOAT,
                output_layout=types.NCHW)
            
            # 转置为CHW格式
            # images = fn.transpose(images, perm=[2, 0, 1], device="gpu")
            
            return images.gpu(), labels.gpu()
        
        # 创建pipeline
        pipe = train_pipeline(
            batch_size=batch_size,
            num_threads=num_workers,
            device_id=self.gpu,
            seed=42 + self.gpu,
            prefetch_queue_depth=2,
            )
        
        # 创建DALI迭代器
        train_loader = DALIClassificationIterator(
            pipe,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True)
        
        return train_loader

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('data.batch_size')
    @param('training.distributed')
    def create_train_loader(self, train_dataset, num_workers, batch_size, distributed):
        @pipeline_def(batch_size=batch_size,
                      num_threads=4,           # GPU pipeline 线程
                      device_id=self.gpu,
                      seed=42 + self.gpu,
                      py_num_workers=num_workers,   # 负责磁盘 IO
                      py_start_method='spawn',
                      prefetch_queue_depth=2)
        def train_pipe():
            jpegs, labels = fn.readers.file(
                file_root=train_dataset,
                random_shuffle=True,
                name='Reader')

            # 1. GPU decode + random crop
            images = fn.decoders.image_random_crop(
                jpegs,
                device='mixed',                 # GPU
                output_type=types.RGB,
                random_area=[0.08, 1.0],
                random_aspect_ratio=[0.75, 1.333333])

            # 2. 随机水平翻转
            images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))

            # 3. resize 到目标分辨率（可以动态 epoch 调节）
            images = fn.resize(
                images,
                resize_x=224,
                resize_y=224,
                device='gpu')

            # 4. normalize & NCHW
            images = fn.crop_mirror_normalize(
                images,
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                scale=1. / 255,
                output_layout=types.NCHW,
                device='gpu')

            return images, labels.gpu()

        pipe = train_pipe()
        # DROP 最后一组不完整的 batch，防止 BN 出错
        loader = dalitorch.DALIClassificationIterator(
            pipe,
            reader_name='Reader',
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True)
        return loader

    # ---------- 验证 loader ----------
    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size, resolution, distributed):
        @pipeline_def(batch_size=batch_size,
                      num_threads=4,
                      device_id=self.gpu,
                      seed=42 + self.gpu,
                      py_num_workers=num_workers,
                      py_start_method='spawn',
                      prefetch_queue_depth=2)
        def val_pipe():
            jpegs, labels = fn.readers.file(
                file_root=val_dataset,
                random_shuffle=False,
                name='Reader')

            # GPU 解码
            images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)

            # resize short side
            images = fn.resize(
                images,
                resize_shorter=int(resolution / DEFAULT_CROP_RATIO),
                device='gpu')

            # center crop
            images = fn.crop_mirror_normalize(
                images,
                crop=(resolution, resolution),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                scale=1. / 255,
                output_layout=types.NCHW,
                device='gpu')

            return images, labels.gpu()

        pipe = val_pipe()
        loader = dalitorch.DALIClassificationIterator(
            pipe,
            reader_name='Reader',
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True)
        return loader

    @param('training.epochs')
    @param('logging.log_level')
    def train(self, epochs, log_level):
        for epoch in range(epochs):
            res = self.get_resolution(epoch)
            train_loss = self.train_loop(epoch)

            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
                    'epoch': epoch
                }

                self.eval_and_log(extra_dict)

        self.eval_and_log({'epoch':epoch})
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')
    
    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats['top_5'],
                'val_time': val_time
            }, **extra_dict))

        return stats

    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('training.use_blurpool')
    def create_model_and_scaler(self, arch, pretrained, distributed, use_blurpool):
        scaler = GradScaler()
        model = getattr(models, arch)(pretrained=pretrained)
        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler

     # ---------- 训练循环 ----------
    @param('logging.log_level')
    def train_loop(self, epoch, log_level):
        model = self.model
        model.train()
        losses = []

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        for ix, data in enumerate(tqdm(self.train_loader, total=iters)):
            images = data[0]['data']          # 已在 GPU
            target = data[0]['label'].squeeze(-1).long()

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = model(images)
                loss_train = self.loss(output, target)

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if log_level > 0:
                losses.append(loss_train.detach())
        return ch.stack(losses).mean().item()

     # ---------- 验证循环 ----------
    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for data in tqdm(self.val_loader):
                    images = data[0]['data']        # 已在 GPU
                    target = data[0]['label'].squeeze(-1).long()

                    output = model(images)
                    if lr_tta:
                        output += model(ch.flip(images, dims=[3]))

                    self.val_meters['top_1'](output, target)
                    self.val_meters['top_5'](output, target)
                    self.val_meters['loss'](self.loss(output, target))

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        for m in self.val_meters.values():
            m.reset()
        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=1000,top_k=5).to(self.gpu),
            'loss': MeanScalarMetric().to(self.gpu)
        }

        if self.gpu == 0:
            folder = (Path(folder) / str(self.uid)).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)

    def log(self, content):
        print(f'=> Log: {content}')
        if self.gpu != 0: return
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count

def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Pytorch imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":
    make_config()
    ImageNetTrainer.launch_from_args()