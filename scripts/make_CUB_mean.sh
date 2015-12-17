#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=data/CUB_200_2011/lmdb
DATA=data/CUB_200_2011
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/CUB_train_lmdb \
  $DATA/CUB_mean.binaryproto

echo "Done."
