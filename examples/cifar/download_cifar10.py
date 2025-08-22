import os
from PIL import Image
from torchvision import datasets
from tqdm import tqdm

def save_cifar_as_images(root="cifar10_raw", out_dir="cifar10_images", split="train", fmt="png"):
    """split: 'train' or 'test'"""
    ds = datasets.CIFAR10(root=root, train=(split == "train"), download=True)
    out_split_dir = os.path.join(out_dir, split)
    os.makedirs(out_split_dir, exist_ok=True)

    # 建立标签文件夹 0~9
    for label in range(10):
        os.makedirs(os.path.join(out_split_dir, str(label)), exist_ok=True)

    for idx, (img, label) in enumerate(tqdm(ds, desc=f"Saving {split}")):
        # img 是 PIL.Image
        label_dir = os.path.join(out_split_dir, str(label))
        img_path = os.path.join(label_dir, f"{idx:05d}.{fmt}")
        img.save(img_path)

# 下载并保存
save_cifar_as_images(split="train")
save_cifar_as_images(split="test")