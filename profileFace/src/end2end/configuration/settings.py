#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : settings.py
# @time    : 7/20/20 4:09 PM
# @desc    : 
'''
configurations = {
    1: dict(
        SEED=1337,  # random seed for reproduce results

        # DATA_ROOT='/dev/shm',
        DATA_ROOT='/home/shengyang/haige_dataset/face_occusion',
        # DATA_NAME='imgs_subset8000', # ms1m 8000个人脸的子集,用于训练前调试
        # DATA_NAME='imgs_subset500',  # ms1m 500个人脸的子集,用于训练前调试
        # DATA_NAME='imgs',            # ms1m人脸
        DATA_NAME='imgs_glintasia',    # deepglint 亚洲人脸
        # the parent root where your train/val/test data are stored
        MODEL_ROOT='./model',  # the root to buffer your checkpoints
        # COMMENTS='IRSE101_ms1m+asia_lr0.02_arcface_bs128x4',  # the log comments
        COMMENTS='IRECA50_ms1m_asia_lr1e-4_sgd_bs128x8',  # the log comments

        # the root to resume training from a saved checkpoint
        # BACKBONE_RESUME_ROOT='',
        # BACKBONE_RESUME_ROOT='model/2020-07-15-09-32_IR_ECA_50_Epoch_10_LOSS_0.825.pth',
        BACKBONE_RESUME_ROOT='model/2020-07-17-06-04_IR_ECA_50_Epoch_60_LOSS_0.962.pth',
        # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT='',

        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152',
        # 'IR_SE_50', 'IRSE_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        # MobileNetV2
        BACKBONE_NAME='IR_ECA_50',

        # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax', 'CircleLoss']
        HEAD_NAME='ArcFace',

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],

        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=128,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
        MOMENTUM=0.9,
        LR=0.0001,  # initial LR
        NUM_EPOCH=20,  # total epoch number
        WARN_UP_EPOCH=2,
        STAGES={10: 0.1,
                15: 0.1},  # epoch stages to decay learning rate
        OPEN_HEAD_LAYERS_EPOCH=10,
        CONTINUE_EPOCH=0,    # continue epoch

        NUM_WORKERS=2,
        EVAL_EPOCH=5,
        GPU_IDS=[0, 1, 2, 3, 4, 5, 6, 7]
    ),
}