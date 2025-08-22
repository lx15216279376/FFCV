#!/usr/bin/env python3
"""
cifar10_to_tfrecord.py
把 CIFAR-10 原始图片目录 → TFRecord(.tfrec) 分片
用法：
    python cifar10_to_tfrecord.py \
        --root ./cifar10_images \
        --out_dir ./cifar10_tfrec \
        --split train \
        --num_shards 10 \
        --compress gzip
"""
import os
import argparse
from pathlib import Path
import multiprocessing as mp
from typing import List, Tuple, Any

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ---------- protobuf helpers ----------
def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# ---------- single shard writer ----------
def write_shard(args: Tuple[int, int, List[Path], List[int], str, bool]):
    shard_id, num_shards, paths, labels, out_tmpl, compress = args
    options = tf.io.TFRecordOptions(compression_type='GZIP' if compress else '')
    out_file = out_tmpl.format(shard_id, num_shards)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with tf.io.TFRecordWriter(out_file, options=options) as writer:
        for path, label in zip(paths, labels):
            try:
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    img_bytes = tf.io.encode_jpeg(np.array(img)).numpy()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': _bytes_feature(img_bytes),
                    'label': _int64_feature(label),
                }))
                writer.write(example.SerializeToString())
            except Exception as e:
                print(f"Skip {path}: {e}")

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        help='CIFAR-10 root containing train/ val/')
    parser.add_argument('--split', choices=['train', 'test'], required=True)
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for .tfrec')
    parser.add_argument('--num_shards', type=int, default=10)
    parser.add_argument('--compress', choices=['gzip', 'none'], default='gzip')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    src_dir = Path(args.root) / args.split
    assert src_dir.exists(), f"{src_dir} not found"

    # build (path, label) list
    cls_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])
    cls_to_idx = {d.name: idx for idx, d in enumerate(cls_dirs)}
    samples = []
    for cls_dir in cls_dirs:
        label = cls_to_idx[cls_dir.name]
        for img_path in cls_dir.glob('*'):
            if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                samples.append((img_path, label))
    print(f'{args.split}: {len(samples)} samples, {len(cls_dirs)} classes')

    # split into shards
    num_shards = args.num_shards
    shard_size = (len(samples) + num_shards - 1) // num_shards
    tasks = []
    out_tmpl = str(Path(args.out_dir) / args.split /
                   f'{args.split}-{{:02d}}-of-{{:02d}}.tfrec')
    compress = args.compress == 'gzip'

    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, len(samples))
        paths, labels = zip(*samples[start:end])
        tasks.append((shard_id, num_shards, paths, labels, out_tmpl, compress))

    with mp.Pool(args.num_workers) as pool:
        list(tqdm(pool.imap(write_shard, tasks), total=len(tasks)))

if __name__ == '__main__':
    main()