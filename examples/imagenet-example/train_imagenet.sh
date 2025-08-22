# Required environmental variables for the script:
export IMAGENET_DIR=/mnt/3fs/datasets/ILSVRC2012_img_train/
export WRITE_DIR=/mnt/3fs/datasets/ffcv_imagenet/

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded
# - quality=90 JPEGs
sudo ./ffcv-imagenet-main/write_imagenet.sh 500 0.50 90