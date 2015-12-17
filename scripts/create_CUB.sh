#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=data/CUB_200_2011/lmdb/
DATA=data/CUB_200_2011/
TOOLS=build/tools/
IMAGES=data/CUB_200_2011/images/
#TRAIN_DATA_ROOT=/path/to/imagenet/train/
#VAL_DATA_ROOT=/path/to/imagenet/val/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi


echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $IMAGES \
    $DATA/all_train.txt \
    $EXAMPLE/CUB_all_train_lmdb

#GLOG_logtostderr=1 $TOOLS/convert_imageset \
#    --resize_height=$RESIZE_HEIGHT \
#    --resize_width=$RESIZE_WIDTH \
#    --shuffle \
#    $IMAGES \
#    $DATA/val.txt \
#    $EXAMPLE/CUB_val_lmdb

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $IMAGES \
    $DATA/all_test.txt \
    $EXAMPLE/CUB_test_lmdb

echo "Done."
