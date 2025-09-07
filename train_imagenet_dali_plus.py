"""
train_imagenet_dali.py
使用 NVIDIA DALI 进行 ImageNet 训练的完整示例
支持：分布式、动态分辨率、AMP、BlurPool 等全部原有功能
"""

import torch as ch
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from torchvision import models
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

# ===== DALI =====
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# ------------------------------------------------------------------ #
# -------------------------- 参数配置 ------------------------------- #
# ------------------------------------------------------------------ #

# 统一分辨率常量
CROP_SIZE  = 224   # 训练 & 最终验证 crop
VAL_SIZE   = 256   # 验证时先 resize 的短边

# 统计cpu内存使用情况，保存路径，每100个batch记录一次
LOG_FILE = '/home/liuxuan/FFCV-main/examples/imagenet-example/ffcv-imagenet-main/log/Cpu_Memory_Status/dali_00.log'          # 可自行修改路径

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
IMAGENET_STD  = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256

# ------------------------------------------------------------------ #
# ------------------------ 学习率/分辨率工具 ------------------------- #
# ------------------------------------------------------------------ #
@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0
    return step_ratio ** (epoch // step_length) * lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

# ------------------------------------------------------------------ #
# --------------------------- 网络组件 ------------------------------ #
# ------------------------------------------------------------------ #
class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        filt = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], dtype=ch.float32) / 16.
        filt = filt.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        x = F.conv2d(x, self.blur_filter, stride=1, padding=1,
                     groups=self.conv.in_channels, bias=None)
        return self.conv(x)

# ------------------------------------------------------------------ #
# -------------------------- DALI pipeline -------------------------- #
# ------------------------------------------------------------------ #
@pipeline_def
def create_train_pipeline(data_dir, shard_id=0, num_shards=1, res=CROP_SIZE):
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=True,
        name="Reader")

    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.random_resized_crop(images, size=res, random_area=[0.08, 1.25])
    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD)

    return images, labels

@pipeline_def
def create_val_pipeline(data_dir, shard_id=0, num_shards=1, crop=CROP_SIZE, val_resize=VAL_SIZE):
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=False,
        name="Reader")

    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.resize(images, resize_shorter=val_resize,
                       interp_type=types.INTERP_TRIANGULAR)
    images = fn.crop_mirror_normalize(
        images,
        crop=(crop, crop),
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD)

    return images, labels

# ------------------------------------------------------------------ #
# ------------------------- DALI 包装器 ----------------------------- #
# ------------------------------------------------------------------ #
class DALIDataloader(DALIGenericIterator):
    def __init__(self, pipeline, reader_name, output_map=["data", "label"], auto_reset=True):
        super().__init__(
            pipelines=pipeline,
            reader_name=reader_name,
            auto_reset=auto_reset,
            output_map=output_map,
            last_batch_policy=LastBatchPolicy.DROP
        )

    def __next__(self):
        data = super().__next__()[0]
        images = data["data"]
        labels = data["label"].squeeze(-1).long()
        labels = labels.cuda(non_blocking=True)
        return images, labels

# ------------------------------------------------------------------ #
# --------------------------- 训练器 ------------------------------- #
# ------------------------------------------------------------------ #
class ImageNetTrainer:
    @param('training.distributed')
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu
        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader   = self.create_val_loader()
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

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        return {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr
        }[lr_schedule_type](epoch)

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
        return int(np.round(interp[0] / 32)) * 32

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, momentum, optimizer, weight_decay, label_smoothing):
        assert optimizer == 'sgd'
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if 'bn' in k]
        other_params = [v for k, v in all_params if 'bn' not in k]
        param_groups = [
            {'params': bn_params, 'weight_decay': 0.},
            {'params': other_params, 'weight_decay': weight_decay}
        ]
        self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ------------------ 训练 DALI loader ------------------ #
    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    def create_train_loader(self, train_dataset, num_workers, batch_size, distributed):
        world_size = dist.get_world_size() if distributed else 1
        rank       = dist.get_rank()       if distributed else 0

        res = self.get_resolution(epoch=0)  # 初始分辨率
        pipe = create_train_pipeline(
            batch_size=batch_size,
            num_threads=num_workers, # DALI 内部的解码线程
            device_id=self.gpu,
            data_dir=train_dataset,
            shard_id=rank,
            num_shards=world_size, 
            prefetch_queue_depth=2,  # 提前准备两个 batch
            res=res,  # 初始分辨率，后续动态改
            seed=42 + rank,
            exec_async=True,
            exec_pipelined=True
        )
        pipe.build()

        return DALIDataloader(pipe, reader_name="Reader")

    # ------------------ 验证 DALI loader ------------------ #
    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size, resolution, distributed):
        world_size = dist.get_world_size() if distributed else 1
        rank       = dist.get_rank()       if distributed else 0

        pipe = create_val_pipeline(
            batch_size=batch_size,
            num_threads=num_workers,
            device_id=self.gpu,
            data_dir=val_dataset,
            shard_id=rank,
            num_shards=world_size,
            crop=resolution,
            val_resize=int(resolution / DEFAULT_CROP_RATIO),
            seed=42 + rank
        )
        pipe.build()
        return DALIDataloader(pipe, reader_name="Reader")

    # ------------------ 模型 & 其它 ------------------ #
    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('training.use_blurpool')
    def create_model_and_scaler(self, arch, pretrained, distributed, use_blurpool):
        scaler = GradScaler()
        model = getattr(models, arch)(pretrained=pretrained)
        if use_blurpool:
            def apply_blurpool(m):
                for n, c in m.named_children():
                    if isinstance(c, ch.nn.Conv2d) and np.max(c.stride) > 1 and c.in_channels >= 16:
                        setattr(m, n, BlurPoolConv2d(c))
                    else:
                        apply_blurpool(c)
            apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last).to(self.gpu)
        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
        return model, scaler

    # ------------------ 训练/验证循环 ------------------ #
    @param('training.epochs')
    @param('logging.log_level')
    def train(self, epochs, log_level):
        for epoch in range(epochs):
            train_loss = self.train_loop(epoch)
            if log_level > 0:
                self.eval_and_log({'train_loss': train_loss, 'epoch': epoch})
        self.eval_and_log({'epoch': epoch})
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')

    def eval_and_log(self, extra_dict=None):
        extra_dict = extra_dict or {}
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats['top_5'],
                'val_time': val_time,
                **extra_dict
            })
        return stats

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

    @param('logging.log_level')
    def train_loop(self, epoch, log_level):
        model = self.model
        model.train()
        losses = []

        # 动态分辨率：重建 train_loader（可选）
        crop = self.get_resolution(epoch)
        self.train_loader = self.create_train_loader()  # 会用到新的 crop

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        for ix, (images, target) in enumerate(tqdm(self.train_loader, total=iters)):
            self.log_system_stats(ix)   # 记录系统资源使用情况
            # images/target 已经是 GPU Tensor
            for pg in self.optimizer.param_groups:
                pg['lr'] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = model(images)
                loss = self.loss(output, target)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if log_level > 0:
                losses.append(loss.detach())

        return ch.stack(losses).mean().item() if losses else ch.tensor(0.)

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()
        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
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

    # ------------------ 日志 ------------------ #
    @param('logging.folder')
    def initialize_logger(self, folder):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric().to(self.gpu)
        }
        if self.gpu == 0:
            folder = (Path(folder) / str(self.uid)).absolute()
            folder.mkdir(parents=True, exist_ok=True)
            self.log_folder = folder
            self.start_time = time.time()
            print(f'=> Logging in {self.log_folder}')
            with open(folder / 'params.json', 'w') as f:
                json.dump({'.'.join(k): self.all_params[k] for k in self.all_params.entries}, f, default=str)

    def log(self, content):
        if self.gpu != 0:
            return
        cur = time.time()
        with open(self.log_folder / 'log', 'a') as f:
            f.write(json.dumps({'timestamp': cur, 'relative_time': cur - self.start_time, **content}) + '\n')

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

# ------------------------------------------------------------------ #
# --------------------------- 工具类 -------------------------------- #
# ------------------------------------------------------------------ #
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state('sum',  default=ch.tensor(0.),  dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0),   dist_reduce_fx='sum')

    def update(self, x):
        self.sum += x.sum()
        self.count += x.numel()

    def compute(self):
        return self.sum / self.count

# ------------------------------------------------------------------ #
# --------------------------- 启动 --------------------------------- #
# ------------------------------------------------------------------ #
def make_config(quiet=False):
    cfg = get_current_config()
    parser = ArgumentParser(description='ImageNet training with DALI')
    cfg.augment_argparse(parser)
    cfg.collect_argparse_args(parser)
    cfg.validate(mode='stderr')
    if not quiet:
        cfg.summary()

if __name__ == "__main__":
    make_config()
    ImageNetTrainer.launch_from_args()