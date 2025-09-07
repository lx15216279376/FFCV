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
import psutil
import csv
import json
from uuid import uuid4
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

import webdataset as wds
from torchvision import transforms

# 统计cpu内存使用情况，保存路径，每100个batch记录一次
LOG_FILE = '/home/liuxuan/FFCV-main/examples/imagenet-example/ffcv-imagenet-main/log/Cpu_Memory_Status/wbs_00.log'          # 可自行修改路径

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

# ... (其他函数保持不变)
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
    # ... (其他方法保持不变)
    @param('training.distributed')
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())
        self.distributed = distributed

        if distributed:
            self.setup_distributed()

        self.train_loader,self.train_steps = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.initialize_logger()
        # 记录系统资源使用情况
        self._proc = psutil.Process(os.getpid())
        self._proc.cpu_percent(None)  # 先采样一次，丢掉结果
        

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

    # 根据epoch获取学习率
    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr
        }

        return lr_schedules[lr_schedule_type](epoch)
    
    # 根据epoch计算当前输入分辨率
    # resolution tools
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

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    # 构造优化器
    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing):
        assert optimizer == 'sgd'

        # Only do weight decay on non-batchnorm parameters
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

    @param('data.train_dataset')   # 现在传入的是 WebDataset 的 tar 列表或路径模板
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    def create_train_loader(self, train_dataset, num_workers, batch_size, distributed):
        # 1. 数据增强 / 归一化
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # 2. 构造 WebDataset
        # 计算每卡实际 step 数
        # print(f"[DEBUG] batch_size = {batch_size}")
        world_size  = dist.get_world_size() if distributed else 1
        # print(f"[DEBUG] world_size = {world_size}")
        real_batch_size  = batch_size  # 全局 batch
        per_gpu_batch_size = real_batch_size // world_size
        # print(f"[DEBUG] per_gpu_batch = {per_gpu_batch_size}")
        num_samples = 1281167
        per_gpu_steps = (num_samples // world_size + per_gpu_batch_size - 1) // per_gpu_batch_size

        urls = train_dataset                    # 如 "/path/imagenet-train-{000000..001279}.tar"
        dataset = (wds.WebDataset(urls,resampled=True)  # 无限循环
                .decode('pil')              # 解码为 PIL
                .to_tuple('jpg', 'cls')     # 返回 (img_tensor, label_int)
                .map_tuple(train_transform, int)
                .with_length(per_gpu_steps))  # 每 epoch 固定步数

        # 3. 分布式采样
        if distributed:
            dataset = dataset.subset(lambda x: x % dist.get_world_size() == dist.get_rank())


        # 4. DataLoader
        loader = wds.WebLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False               # WebDataset 内部已处理采样
        )

        return loader,per_gpu_steps

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size, resolution, distributed):
        val_transform = transforms.Compose([
            transforms.Resize(int(resolution / DEFAULT_CROP_RATIO)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        urls = val_dataset                      # 如 "/path/imagenet-val-{000000..000049}.tar"
        dataset = (wds.WebDataset(urls)
                .decode('pil')
                .to_tuple('jpg', 'cls')
                .map_tuple(val_transform, int))

        if distributed:
            dataset = dataset.subset(lambda x: x % dist.get_world_size() == dist.get_rank())

        loader = wds.WebLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False
        )
        return loader

# ... (其他代码保持不变)
     # 主训练循环
    @param('training.epochs')
    @param('logging.log_level')
    def train(self, epochs, log_level):
        for epoch in range(epochs):
            res = self.get_resolution(epoch)
            # self.decoder.output_size = (res, res)
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
    
    # 验证并写日志
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

    # 构造模型+混合精度scaler
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

    def log_system_stats(self,step):
        """每 100 个 batch 记录一次系统资源"""
        if step % 100 != 0:
            return
        # proc = psutil.Process(os.getpid())
        mem_mb = self._proc.memory_info().rss / 1024 ** 2
        cpu_pct = self._proc.cpu_percent(None)
        ts = time.time()                   # Unix 时间戳，后续可转日期
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            # 如果文件为空，先写表头
            if f.tell() == 0:
                writer.writerow(['step', 'timestamp', 'cpu_percent', 'mem_mb'])
            writer.writerow([step, ts, cpu_pct, mem_mb])

    # 训练循环，一个epoch
    @param('data.batch_size')
    @param('logging.log_level')
    def train_loop(self, epoch, log_level,batch_size):
        model = self.model
        model.train()
        losses = []

        # 步数手动计算
        # num_samples = 1281167
        # if self.distributed:
        #     num_samples //= dist.get_world_size()
        # steps_per_epoch = (num_samples + batch_size - 1) // batch_size
        steps_per_epoch = self.train_steps
        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        lrs = np.interp(np.arange(steps_per_epoch), [0, steps_per_epoch],
                        [lr_start, lr_end])

        iterator = tqdm(self.train_loader, total=steps_per_epoch, desc=f'Epoch {epoch}')
        for ix, (images, target) in enumerate(iterator):
            self.log_system_stats(ix)   # 记录系统资源使用情况
            images = images.to(self.gpu, non_blocking=True)
            target = target.to(self.gpu, non_blocking=True)

            lr_idx = min(ix, len(lrs) - 1)      # <-- 防止越界
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[lr_idx]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = self.model(images)
                loss_train = self.loss(output, target)

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end

            ### Logging start
            if log_level > 0:
                losses.append(loss_train.detach())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss_train.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)
            ### Logging end

    # 验证循环
    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    images = images.to(self.gpu, non_blocking=True)
                    target = target.to(self.gpu, non_blocking=True)
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)

                    loss_val = self.loss(output, target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    # 初始化日志记录器
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

    # 多进程启动入口
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

# Utils
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

# Running
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