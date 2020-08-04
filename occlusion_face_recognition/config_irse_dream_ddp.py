#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com && dylenzheng@gmail.com
# @IDE: PyCharm
# @File: config_ori_ddp.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.10.21 16:17    samon      v0.1        creation
# 2020.07.24 09:37    dylen      v0.2        revision for profileFace

configurations = {
    1: dict(
        SEED=1337,  # random seed for reproduce results

        # DATA_ROOT='/dev/shm',
        # DATA_NAME='imgs_subset8000',
        # DATA_NAME='imgs_subset500',
        # DATA_NAME='profileAsia',
        TRAIN_DATA_DIR='/home/zhex/data/profiledataset/profileAsia/train',
        TRAIN_IMAGE ="/home/zhex/data/profiledataset/trainImage.txt",
        TRAIN_LABEL = "/home/zhex/data/profiledataset/trainLabel.txt",

        VAL_DATA_DIR = "/home/zhex/data/evoLVe_val_dataset",
        YAW_BLP_DIR = "/home/zhex/data/evoLVe_val_dataset/yaw_blp_npy",
        YAW_IMG_DIR = "/home/zhex/data/evoLVe_val_dataset/yaw_image_npy",

        # the parent root where your train/val/test data are stored
        MODEL_ROOT='./model',  # the root to buffer your checkpoints
        # COMMENTS='IRSE101_ms1m+asia_lr0.02_arcface_bs128x4',  # the log comments
        COMMENTS='IRECA50_ms1m_lr1e-3_sgd_bs128x8',  # the log comments

        # the root to resume training from a saved checkpoint
        # BACKBONE_RESUME_ROOT='',
        BACKBONE_RESUME_ROOT='model/2020-07-24-20-42_IR_SE_DREAM_101_Epoch_23_LOSS_0.041.pth',
        # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT='model/2020-07-24-20-42_ArcFace_Epoch_23_LOSS_0.041.pth',

        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152',
        # 'IR_SE_50', 'IRSE_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152',
        # 'IR_SE_DREAM_101']
        BACKBONE_NAME='IR_101',

        # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax', 'CircleLoss']
        HEAD_NAME='ArcFace',

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],

        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=32,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
        MOMENTUM=0.9,
        LR=0.001,  # initial LR
        NUM_EPOCH=100,  # total epoch number
        WARN_UP_EPOCH=5,
        STAGES={15: 0.1,
                30: 0.1,
                45: 0.1},  # epoch stages to decay learning rate
        OPEN_HEAD_LAYERS_EPOCH=0,
        CONTINUE_EPOCH=0,    # continue epoch

        FREQUENCY = 5,
        NUM_WORKERS=2,
        EVAL_EPOCH=5,
        GPU_IDS=[0]
    ),
}
