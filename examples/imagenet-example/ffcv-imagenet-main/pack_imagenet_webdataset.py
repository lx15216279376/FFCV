#!/usr/bin/env python3
"""
imagenet2wds.py
ImageNet 原始图片目录 → WebDataset (.tar)
"""
import os, webdataset as wds
from tqdm import tqdm

# ========== 改成自己的路径 ==========
IMAGENET_ROOT = '/home/lenovo/ffcv/ffcv-main/examples/imagenet-example/ffcv-imagenet-main/datasets'      # 含 train/  val/
OUT_DIR       = '/mnt/3fs/datasets/webdataset_imagenet'  # 输出目录
MAX_PER_SHARD = 5000                     # 每个 .tar 最多多少张
# ====================================

os.makedirs(OUT_DIR, exist_ok=True)

def synset_to_idx(src_dir):
    """返回 dict：{'n01440764': 0, 'n01443537': 1, ...}"""
    return {d: idx for idx, d in enumerate(sorted(os.listdir(src_dir)))}

# 如果本地没有 synset_words.txt，可手动建一个；或把 label 直接用字符串。
# synset2idx = synset_to_idx(os.path.join(IMAGENET_ROOT, 'train'))

def pack_split(split):
    src_dir = os.path.join(IMAGENET_ROOT, split)
    synset2idx = synset_to_idx(src_dir)   # 用文件夹顺序
    pattern = os.path.join(OUT_DIR, f'imagenet-{split}-%06d.tar')
    with wds.ShardWriter(pattern, maxcount=MAX_PER_SHARD) as sink:
        for synset in os.listdir(src_dir):
            label = synset2idx.get(synset, -1)
            if label < 0:               # 未知类别跳过
                continue
            cls_dir = os.path.join(src_dir, synset)
            for fn in tqdm(os.listdir(cls_dir), desc=f'{split}/{synset}', leave=False):
                if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(cls_dir, fn)
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                key = os.path.splitext(fn)[0]
                sink.write({
                    "__key__": key,
                    "jpg" if fn.lower().endswith(('.jpg', '.jpeg')) else "png": img_bytes,
                    "cls": label
                })

if __name__ == '__main__':
    for s in ('ILSVRC2012_img_train', 'ILSVRC2012_val'):
        pack_split(s)
    print(f"✅ 已写入 {OUT_DIR}")