#!/usr/bin/env bash
model_dir=model
if [ ! -d $model_dir ]; then
    mkdir $model_dir
fi
log_dir=log
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

curr_date=$(date +'%m_%d_%H_%M') 
log_file="./log/$curr_date.log"

# train the model with GPUs 0, 1, 2, and 3
CUDA_VISIBLE_DEVICES=0 python main.py  \
    --end2end   --img_dir /home/zhex/data/msceleb/image \
    2>&1 | tee $log_file
