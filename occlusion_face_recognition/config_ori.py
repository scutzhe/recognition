#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: config_ori.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.10.21 16:17    samon      v0.1        creation

import torch

configurations = {
    1: dict(
        SEED=1337,  # random seed for reproduce results

        # DATA_ROOT='/dev/shm',
        DATA_ROOT='/home/shengyang/haige_dataset/face_occusion',
        # the parent root where your train/val/test data are stored
        MODEL_ROOT='./model',  # the root to buffer your checkpoints
        LOG_ROOT='./log',  # the root to log your train/val status
        BACKBONE_RESUME_ROOT = './model/Backbone_IR_SE_101_Epoch_3_Batch_68232_Time_2019-11-29-21-01_checkpoint.pth', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = './model/Head_ArcFace_Epoch_3_Batch_68232_Time_2019-11-29-21-01_checkpoint.pth', # the root to resume training from a saved checkpoint
        # BACKBONE_RESUME_ROOT='.',  # the root to resume training from a saved checkpoint
        # HEAD_RESUME_ROOT='.',  # the root to resume training from a saved checkpoint

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME='ArcFace',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=512,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.001,  # initial LR
        NUM_EPOCH=120,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
        MOMENTUM=0.9,
        STAGES=[20, 60, 90],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=True,
        PIN_MEMORY=True,
        NUM_WORKERS=8,
    ),
}
