#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: train_pw.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.11.14 09:08    shengyang      v0.1        creation

import os
import os.path as osp
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from backbone.model_irse import PW_IR_SE_101
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from util.utils import get_val_data, separate_ir_bn_paras
from util.utils import separate_resnet_bn_paras, perform_val

from config_pw import configurations
from loss.mask_loss import MaskLoss
from util.checkpoint_tools import load_pretrained_weights


if __name__ == '__main__':
    # ======= hyperparameters & data loaders =======#
    cfg = configurations[1]
    BACKBONE_NAME = "IR_SE_101"
    SEED = cfg['SEED']  # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train/val/test data are stored
    MODEL_ROOT = cfg['MODEL_ROOT']  # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT']  # the root to log your train/val status

    if not osp.exists(MODEL_ROOT):
        os.makedirs(MODEL_ROOT)
    if not osp.exists(LOG_ROOT):
        os.makedirs(LOG_ROOT)

    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    MASKSAMPLE_ROOT = cfg['MASKSAMPLE']  # the root to mask sample

    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR']  # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES']  # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU']  # flag to use multiple GPUs
    GPU_ID = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))  # specify your GPU ids
    # GPU_ID = cfg['GPU_ID']  # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp,\
    lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame,\
    calfw_issame, cplfw_issame, vgg2_fp_issame = get_val_data(DATA_ROOT)

    BACKBONE = PW_IR_SE_101(INPUT_SIZE, masksample=None)
    NUM_CLASS = 85738

    HEAD_DICT = {'ArcFace': ArcFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 'CosFace': CosFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 'SphereFace': SphereFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 'Am_softmax': Am_softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID)}
    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    # print(HEAD)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

    LOSS_DICT = {'Focal': FocalLoss(),
                 'Softmax': nn.CrossEntropyLoss()}
    LOSS = LOSS_DICT[LOSS_NAME]
    MASKLOSS = MaskLoss()
    FEATURELOSS = torch.nn.CosineEmbeddingLoss()

    print("=" * 60)
    print(LOSS)
    print("{} Loss Generated".format(LOSS_NAME))
    print("=" * 60)

    if BACKBONE_NAME.find("IR") >= 0:
        # separate batch_norm parameters from others;
        # do not do weight decay for batch_norm parameters to improve the generalizability
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_ir_bn_paras(BACKBONE)
        _, head_paras_wo_bn = separate_ir_bn_paras(HEAD)
    else:
        # separate batch_norm parameters from others;
        # do not do weight decay for batch_norm parameters to improve the generalizability
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE)
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
                           {'params': backbone_paras_only_bn}], lr=LR, momentum=MOMENTUM)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)

    # optionally resume from a checkpoint
    print("=" * 60)
    if osp.isfile(BACKBONE_RESUME_ROOT):
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
        # BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        load_pretrained_weights(BACKBONE, BACKBONE_RESUME_ROOT)
    else:
        print("No Checkpoint Found at '{}'. Please Have a Check or Continue to Train from Scratch".format(
            BACKBONE_RESUME_ROOT))

    if osp.isfile(HEAD_RESUME_ROOT):
        print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
        # HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
        load_pretrained_weights(HEAD, HEAD_RESUME_ROOT)
    else:
        print("No Checkpoint Found at '{}'. Please Have a Check or Continue to Train from Scratch".format(
            HEAD_RESUME_ROOT))
    print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    # perform validation & save checkpoints per epoch
    # validation statistics per epoch (buffer for visualization)
    print("=" * 60)
    print("Perform Evaluation on LFW and Save Checkpoints...")
    accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE,
                                                                      BACKBONE, lfw, lfw_issame)

    accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                               BATCH_SIZE, BACKBONE, cfp_ff,
                                                                               cfp_ff_issame)
    accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                               BATCH_SIZE, BACKBONE, cfp_fp,
                                                                               cfp_fp_issame)
    accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                            BATCH_SIZE, BACKBONE, agedb, agedb_issame)
    accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                            BATCH_SIZE, BACKBONE, calfw, calfw_issame)
    accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                            BATCH_SIZE, BACKBONE, cplfw, cplfw_issame)
    accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                                  BATCH_SIZE, BACKBONE, vgg2_fp,
                                                                                  vgg2_fp_issame)
    print(
            "Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {}, CPLFW Acc: {}, VGG2_FP Acc: {}".format(
                accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, accuracy_calfw,
                accuracy_cplfw, accuracy_vgg2_fp))
    print("=" * 60)
