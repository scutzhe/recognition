#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: config_pw.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.11.14 11:31    shengyang      v0.1        creation

import torch

configurations = {
    1: dict(
        SEED=1337,  # random seed for reproduce results

        # DATA_ROOT='/home/shengyang/haige_dataset/face_dataset2',
        DATA_ROOT='/home/shengyang/haige_dataset/face_occusion',
        # the parent root where your train/val/test data are stored
        MODEL_ROOT='./pw_model',  # the root to buffer your checkpoints
        LOG_ROOT='./pw_log',  # the root to log your train/val status

        BACKBONE_RESUME_ROOT='./pw_model/Backbone_PW_IR_SE_101_Epoch_28_Batch_253708_Time_2019-11-24-09-07_checkpoint.pth',
        # BACKBONE_RESUME_ROOT='.',
        # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT='./pw_model/Head_ArcFace_Epoch_28_Batch_253708_Time_2019-11-24-09-07_checkpoint.pth',

        # mask samples
        MASKSAMPLE='./mask_sample.pkl',
        # the root to resume training from a saved checkpoint

        # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        HEAD_NAME='ArcFace',
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        # BATCH_SIZE=16,
        BATCH_SIZE=640,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        # LR=0.0005,  # initial LR
        LR=0.0002,  # initial LR
        NUM_EPOCH=30,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
        MOMENTUM=0.9,
        STAGES=[10, 20, 35],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=True,
        # flag to use multiple GPUs; if you choose to train with single GPU,
        # you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        # GPU_ID=[0, 1, 2, 3, 4],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=32,
    ),
}
