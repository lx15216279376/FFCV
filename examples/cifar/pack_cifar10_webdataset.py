#!/usr/bin/env python3
"""
pack_cifar10_wds_to_dir.py
本地原始图片 → WebDataset → 指定目录
"""

import os
import webdataset as wds
from tqdm import tqdm

# -------------------------------------------------
# 只需改这两个变量
SRC_DIR      = 'cifar10_images'       # 原始图片根目录
OUT_ROOT_DIR = 'cifar10_wds'        # 你想保存 .tar 的目录
# -------------------------------------------------

os.makedirs(OUT_ROOT_DIR, exist_ok=True)

def pack_split(split: str, max_per_shard: int = 1000):
    src_split_dir = os.path.join(SRC_DIR, split)                 # …/train 或 …/test
    out_pattern   = os.path.join(OUT_ROOT_DIR,
                                 f'cifar10-{split}-%06d.tar')    # 输出到指定目录
    with wds.ShardWriter(out_pattern, maxcount=max_per_shard) as sink:
        for label in os.listdir(src_split_dir):
            label_dir = os.path.join(src_split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fn in tqdm(os.listdir(label_dir),
                           desc=f'{split}/{label}',
                           leave=False):
                if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                key = os.path.splitext(fn)[0]
                img_path = os.path.join(label_dir, fn)
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                sample = {
                    '__key__': key,
                    'png' if fn.endswith('.png') else 'jpg': img_bytes,
                    'cls': int(label)
                }
                sink.write(sample)

if __name__ == '__main__':
    for s in ('train', 'test'):
        pack_split(s)
    print(f'✅ WebDataset 已保存到 {os.path.abspath(OUT_ROOT_DIR)}')