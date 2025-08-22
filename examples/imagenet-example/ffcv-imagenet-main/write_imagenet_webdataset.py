import os, json, webdataset as wds
from PIL import Image
from tqdm import tqdm

# 1. 原始数据目录结构假设
# images/
#   ├── cat/
#   │   ├── 1.jpg
#   │   └── ...
#   └── dog/
#       ├── 1.jpg
#       └── ...

root = "images"                 # 你的原始数据根目录
out_pattern = "train-%06d.tar"  # 输出分片文件名模板
maxcount = 1000                 # 每个分片最多多少条样本

# 2. 生成样本迭代器
def sample_iter():
    idx = 0
    for label in os.listdir(root):
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        for file in os.listdir(label_dir):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(label_dir, file)
            base = f"{idx:08d}"          # 全局唯一 basename
            idx += 1

            with open(img_path, "rb") as f:
                img_bytes = f.read()
            label_txt = label            # 这里把文件夹名当标签
            # 如果有更多模态，继续往字典里塞即可
            yield {
                "__key__": base,
                "jpg": img_bytes,
                "txt": label_txt,
                # "json": json.dumps({"extra": 123})
            }

# 3. 写入 WebDataset（自动分片）
with wds.ShardWriter(out_pattern, maxcount=maxcount) as sink:
    for sample in tqdm(sample_iter(), desc="Packing"):
        sink.write(sample)