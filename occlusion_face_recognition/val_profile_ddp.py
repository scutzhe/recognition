#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: train_ori.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.10.21 16:09    samon      v0.1        creation
# 2020.07.24 08:35    dylen      v0.2        revision for profileFace

import os
import time
from config_val_ddp import configurations

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, configurations[1]["GPU_IDS"]))

import torch
import torch.nn as nn

from backbone.model_irse import IR_SE_50, IR_SE_101, IR_ECA_50, IR_ECA_101
from backbone.model_dream import IR_SE_DREAM_101
from util.utils import get_val_pair_yaw ,perform_val_yaw
from util.checkpoint_tools import load_pretrained_weights

from tensorboardX import SummaryWriter
import os
import os.path as osp
import logging
from datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def val():
    cfg = configurations[1]
    torch.manual_seed(cfg['SEED'])   # random seed for reproduce results
    valDataDir = cfg["VAL_DATA_DIR"]
    model_root = cfg['MODEL_ROOT']
    # support: ResNet_50, ResNet_101, ResNet_152, IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
    backbone_name = cfg['BACKBONE_NAME']
    head_name = cfg['HEAD_NAME']

    input_size = cfg['INPUT_SIZE']
    rgb_mean = cfg['RGB_MEAN']  # for normalize inputs
    rgb_std = cfg['RGB_STD']
    embedding_size = cfg['EMBEDDING_SIZE']  # feature dimension
    batch_size = cfg['BATCH_SIZE']
    drop_last = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR']  # initial LR
    num_epoch = cfg['NUM_EPOCH']
    continue_epoch = cfg["CONTINUE_EPOCH"]
    weight_decay = cfg['WEIGHT_DECAY']
    momentum = cfg['MOMENTUM']
    comments = cfg['COMMENTS']
    stages = cfg['STAGES']  # epoch stages to decay learning rate
    open_backbone_epoch = cfg['OPEN_HEAD_LAYERS_EPOCH']  # epoch stages to decay learning rate
    num_workers = cfg['NUM_WORKERS']  # epoch stages to decay learning rate

    backbone_resume_root = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    head_resume_root = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[%(asctime)s]-[pid:%(process)d]-%(message)s")
    # logging.basicConfig(filename=f"./log/{comments}.log", format="[%(asctime)s]-[pid:%(process)d]-%(message)s")

    logger.setLevel(logging.INFO)

    log_root = f"./log/{datetime.now().strftime('%m-%d')}_{comments}"


    logger.info("starting validation...")

    lfw, lfw_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'lfw')
    cfp_ff, cfp_ff_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'cfp_ff')
    cfp_fp, cfp_fp_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'cfp_fp')
    cplfw, cplfw_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'cplfw')
    vgg2_fp, vgg2_fp_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'vgg2_fp')
    agedb, agedb_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'agedb_30')
    calfw, calfw_isSame, yaw = get_val_pair_yaw(valDataDir, "image",'calfw')

    # ======= model & loss & optimizer =======#
    if backbone_name == "IR_SE_DREAM_101":
        backbone = IR_SE_DREAM_101(input_size)
    elif backbone_name == 'IR_SE_101':
        backbone = IR_SE_101(input_size)
    elif backbone_name == 'IR_SE_50':
        backbone = IR_SE_50(input_size)
    elif backbone_name == 'IR_ECA_101':
        backbone = IR_ECA_101(input_size)
    elif backbone_name == 'IR_ECA_50':
        backbone = IR_ECA_50(input_size)
    else:
        raise Exception(f"not support this backbone:{backbone_name}")

    load_pretrained_weights(backbone, backbone_resume_root, map_location="cpu")
    backbone.cuda()

    gpu = 0
    epoch = 0
    acc_lfw, best_threshold_lfw, roc_curve_lfw = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                             embedding_size, batch_size,
                                                             backbone, lfw, lfw_isSame, yaw)
    with SummaryWriter(log_root) as writer:
        writer.add_scalar('{}_Accuracy'.format("LFW"), acc_lfw, epoch+1)
    logger.info("=" * 60)
    logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: LFW: {acc_lfw:.5f}"
                f" Threshold: {best_threshold_lfw:.3f}")
    logger.info("=" * 60)

    acc_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                      embedding_size, batch_size,
                                                                      backbone, cfp_ff, cfp_ff_isSame, yaw)
    with SummaryWriter(log_root) as writer:
        writer.add_scalar('{}_Accuracy'.format("CFP_FF"), acc_cfp_ff, epoch + 1)
    logger.info("=" * 60)
    logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CFP_FF: {acc_cfp_ff:.5f}"
                f" Threshold: {best_threshold_cfp_ff:.3f}")
    logger.info("=" * 60)

    acc_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                      embedding_size, batch_size,
                                                                      backbone, cfp_fp, cfp_fp_isSame, yaw)
    with SummaryWriter(log_root) as writer:
        writer.add_scalar('{}_Accuracy'.format("CFP_FP"), acc_cfp_fp, epoch + 1)
    logger.info("=" * 60)
    logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CFP_FP: {acc_cfp_fp:.5f}"
                f" Threshold: {best_threshold_cfp_fp:.3f}")
    logger.info("=" * 60)

    acc_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                   embedding_size, batch_size,
                                                                   backbone, cplfw, cplfw_isSame, yaw)
    with SummaryWriter(log_root) as writer:
        writer.add_scalar('{}_Accuracy'.format("CPLFW"), acc_cplfw, epoch + 1)
    logger.info("=" * 60)
    logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CPLFW: {acc_cplfw:.5f}"
                f" Threshold: {best_threshold_cplfw:.3f}")
    logger.info("=" * 60)

    acc_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                         embedding_size, batch_size,
                                                                         backbone, vgg2_fp, vgg2_fp_isSame, yaw)
    with SummaryWriter(log_root) as writer:
        writer.add_scalar('{}_Accuracy'.format("VGGFace2_FP"), acc_vgg2_fp, epoch + 1)
    logger.info("=" * 60)
    logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: VGG2_FP: {acc_vgg2_fp:.5f}"
                f" Threshold: {best_threshold_vgg2_fp:.3f}")
    logger.info("=" * 60)

    acc_agedb, best_threshold_agedb, roc_curve_agedb = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                   embedding_size, batch_size,
                                                                   backbone, agedb, agedb_isSame, yaw)
    with SummaryWriter(log_root) as writer:
        writer.add_scalar('{}_Accuracy'.format("AgeDB"), acc_agedb, epoch + 1)
    logger.info("=" * 60)
    logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: AgeDB: {acc_agedb:.5f}"
                f" Threshold: {best_threshold_agedb:.3f}")
    logger.info("=" * 60)

    acc_calfw, best_threshold_calfw, roc_curve_calfw = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                   embedding_size, batch_size,
                                                                   backbone, calfw, calfw_isSame, yaw)
    with SummaryWriter(log_root) as writer:
        writer.add_scalar('{}_Accuracy'.format("CALFW"), acc_calfw, epoch + 1)
    logger.info("=" * 60)
    logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CALFW: {acc_calfw:.5f}"
                f" Threshold: {best_threshold_calfw:.3f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    val()
