from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', required=True),
    epochs=Param(int, 'Number of epochs to run for', required=True),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', required=True),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=8),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, 'Path to training dataset', required=True),
    val_dataset=Param(str, 'Path to validation dataset', required=True),
)

@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(train_dataset=None, val_dataset=None,
                     batch_size=None, num_workers=None):
    """
    目录应包含
    train_dataset/cifar-10-batches-py/
    val_dataset  /cifar-10-batches-py/
    """
    start_time = time.time()
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD  = [0.2023, 0.1994, 0.2010]

    @pipeline_def
    def _pipe(is_training: bool, root: str):
        images, labels = fn.readers.cifar10(
            root=root,
            is_training=is_training,
            random_shuffle=is_training,
            name="Reader"
        )
        images = fn.cast(images, dtype=types.FLOAT)
        images = fn.normalize(
            images,
            mean=CIFAR_MEAN * 255,
            std=CIFAR_STD * 255,
            scale=1 / 255.
        )
        return images, labels

    train_pipe = _pipe(
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=0,
        is_training=True,
        root=train_dataset,
        seed=42
    )
    val_pipe = _pipe(
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=0,
        is_training=False,
        root=val_dataset,
        seed=42
    )
    train_pipe.build(); val_pipe.build()

    return {
        "train": DALIGenericIterator(train_pipe, ["data", "label"], auto_reset=True),
        "test":  DALIGenericIterator(val_pipe,   ["data", "label"], auto_reset=True),
    }, start_time

# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton )
class Mul(ch.nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            ch.nn.ReLU(inplace=True)
    )

# 构建模型
def construct_model():
    num_class = 10
    model = ch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=ch.channels_last).cuda()
    return model

@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.lr_peak_epoch')
def train(model, loaders, lr=None, epochs=None, label_smoothing=None,
          momentum=None, weight_decay=None, lr_peak_epoch=None):
    # 初始化SGD优化器和学习率调度器
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train'])
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # 训练循环
    for _ in range(epochs):
        for data in tqdm(loaders['train']):
            ims = data[0]["data"].cuda(non_blocking=True)
            labs = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims.cuda())
                loss = loss_fn(out, labs.cuda())

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

@param('training.lr_tta')
def evaluate(model, loaders, lr_tta=False):
    model.eval()
    with ch.no_grad():
        for name in ['train', 'test']:
            total_correct, total_num = 0., 0.
            for ims, labs in tqdm(loaders[name]):
                with autocast():
                    out = model(ims.cuda())
                    if lr_tta:
                        out += model(ims.flip(-1).cuda())
                    total_correct += out.argmax(1).eq(labs.cuda()).sum().cpu().item()
                    total_num += ims.shape[0]
            print(f'{name} accuracy: {total_correct / total_num * 100:.1f}%')

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Pytorch CIFAR-10 training')
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    loaders, start_time = make_dataloaders()
    model = construct_model()
    train(model, loaders)
    print(f'Total time: {time.time() - start_time:.5f}')
    evaluate(model, loaders)