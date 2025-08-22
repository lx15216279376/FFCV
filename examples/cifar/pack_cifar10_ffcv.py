#!/usr/bin/env python3
"""
convert_to_beton.py
本地原始图片 → FFCV .beton
"""
import os
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from torchvision.datasets import ImageFolder    # 直接识别类别子文件夹
from torchvision import transforms

# ============ 只需改这两行 ============
SRC_DIR = 'cifar10_images'          # 你的原始图片根目录（含 train/ test/）
OUT_DIR = 'cifar10_ffcv'            # 输出 .beton 的位置
# =====================================

os.makedirs(OUT_DIR, exist_ok=True)

for split in ('train', 'test'):
    dataset = ImageFolder(os.path.join(SRC_DIR, split))  
    out_file = os.path.join(OUT_DIR, f'cifar10_{split}.beton')
    writer = DatasetWriter(out_file, {
        'image': RGBImageField(max_resolution=32, write_mode='raw'),
        'label': IntField()
    })
    writer.from_indexed_dataset(dataset)
    print(f'✅ {out_file} 已生成')