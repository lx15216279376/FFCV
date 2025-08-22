#!/bin/bash
export IMAGENET_DIR=/home/lenovo/ffcv/ffcv-main/examples/imagenet-example/ffcv-imagenet-main/datasets/ILSVRC2012_val
export WRITE_DIR=/mnt/3fs/datasets/ffcv_imagenet/tmp_raw
write_dataset () {
    write_path=$WRITE_DIR/${1}_${2}_${3}_${4}.ffcv
    echo "Writing ImageNet ${1} dataset to ${write_path}"
    sudo /home/lenovo/anaconda3/envs/ffcv/bin/python write_imagenet.py \
        --cfg.dataset=imagenet \
        --cfg.split=${1} \
        --cfg.data_dir=$IMAGENET_DIR \
        --cfg.write_path=$write_path \
        --cfg.max_resolution=${2} \
        --cfg.write_mode=raw \
        --cfg.compress_probability=${3} \
        --cfg.jpeg_quality=$4
}

# write_dataset train $1 $2 $3
write_dataset val $1 $2 $3
