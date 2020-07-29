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
from config_ori_ddp import configurations

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, configurations[1]["GPU_IDS"]))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from backbone.model_irse import IR_SE_50, IR_SE_101, IR_ECA_50, IR_ECA_101
from backbone.model_dream import IR_SE_DREAM_101
from backbone.selfdefine import MsCelebDataset
from head.metrics import ArcFace, CircleLoss
from loss.focal import FocalLoss, FocalLoss2
from util.utils import make_weights_for_balanced_classes, get_val_data, buffer_val
from util.utils import get_val_pair, separate_ir_bn_paras, separate_resnet_bn_paras
from util.utils import warm_up_lr, schedule_lr, perform_val, get_time, AverageMeter, accuracy
from util.utils import get_val_pair_yaw ,perform_val_yaw
from util.checkpoint_tools import load_pretrained_weights, open_specified_layers, open_all_layers, close_all_layers

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import os.path as osp
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import logging
from datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-g', '--gpus', default=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
                        type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    args = parser.parse_args()

    #########################################################
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '192.168.2.251'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    #########################################################


def train(gpu, args):
    torch.backends.cudnn.benchmark = True
    ######################################################################
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.cuda.set_device(gpu)
    ######################################################################

    # ======= hyperparameters & data loaders =======#
    cfg = configurations[1]

    torch.manual_seed(cfg['SEED'])   # random seed for reproduce results

    data_root = cfg['VAL_DATA_ROOT']  # the parent root where your train/val/test data are stored
    # data_name = cfg['DATA_NAME']  # the parent root where your train/val/test data are stored
    trainDataDir = cfg["TRAIN_DATA_DIR"]
    trainImage = cfg["TRAIN_IMAGE"]
    trainLabel = cfg["TRAIN_LABEL"]

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
    warm_up_epoch = cfg['WARN_UP_EPOCH']  # the root to resume training from a saved checkpoint
    eval_epoch = cfg['EVAL_EPOCH']  # the root to resume training from a saved checkpoint
    continue_epoch = cfg['CONTINUE_EPOCH']  # the root to resume training from a saved checkpoint

    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[%(asctime)s]-[pid:%(process)d]-%(message)s")
    # logging.basicConfig(filename=f"./log/{comments}.log", format="[%(asctime)s]-[pid:%(process)d]-%(message)s")

    logger.setLevel(logging.INFO)

    log_root = f"./log/{datetime.now().strftime('%m-%d')}_{comments}"
    if gpu is 0:
        if not osp.exists(log_root):
            os.makedirs(log_root)

    trainTransform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop([input_size[0], input_size[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std),
    ])

    # valTransform = transforms.Compose([
    #     transforms.Resize(input_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=rgb_mean, std=rgb_std),
    # ])

    # dataset_train = datasets.ImageFolder(os.path.join(data_root, data_name), train_transform)
    # load data and prepare dataset
    train_dataset = MsCelebDataset(trainDataDir, trainImage, trainLabel, trainTransform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                            shuffle=True,num_workers=args.workers, pin_memory=True)
    ######################################################################
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                              num_replicas=args.world_size,
                                                              rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=sampler, pin_memory=True, shuffle=False,
                                               num_workers=num_workers, drop_last=drop_last)
    ######################################################################
    # num_class = len(train_dataset)
    num_class = train_dataset.label_num
    print("Number of train classes:{}".format(num_class))

    # val_dataset = MsCelebDataset(valDataDir, valImage, valLabel,valTransform)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
    #                                            pin_memory=True, shuffle=False,
    #                                           num_workers=num_workers, drop_last=drop_last)
    # # valNumClass = len(val_dataset)
    # valNumClass = val_dataset.label_num
    # print("Number of val classes:{}".format(valNumClass))

    # if gpu is 0:
    #     logger.info("Number of Training Classes: {}".format(num_class))
    # if gpu is 0:
    #     lfw, lfw_issame = get_val_pair(data_root, 'lfw')
    # if gpu is 1:
    #     cfp_ff, cfp_ff_issame = get_val_pair(data_root, 'cfp_ff')
    # if gpu is 2:
    #     cfp_fp, cfp_fp_issame = get_val_pair(data_root, 'cfp_fp')
    # if gpu is 3:
    #     cplfw, cplfw_issame = get_val_pair(data_root, 'cplfw')
    # if gpu is 4:
    #     vgg2_fp, vgg2_fp_issame = get_val_pair(data_root, 'vgg2_fp')
    # if gpu is 5:
    #     agedb, agedb_issame = get_val_pair(data_root, 'agedb_30')
    # if gpu is 6:
    #     calfw, calfw_issame = get_val_pair(data_root, 'calfw')

    if gpu is 0:
        logger.info("Number of Training Classes: {}".format(num_class))
    if gpu is 0:
        lfw, lfw_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'lfw')
    if gpu is 1:
        cfp_ff, cfp_ff_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'cfp_ff')
    if gpu is 2:
        cfp_fp, cfp_fp_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'cfp_fp')
    if gpu is 3:
        cplfw, cplfw_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'cplfw')
    if gpu is 4:
        vgg2_fp, vgg2_fp_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'vgg2_fp')
    if gpu is 5:
        agedb, agedb_isSame, yaw = get_val_pair_yaw(valDataDir, "image", 'agedb_30')
    if gpu is 6:
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

    if head_name != 'CircleLoss':
        head = ArcFace(in_features=embedding_size, out_features=num_class)
        loss = FocalLoss()
    else:
        head = CircleLoss(in_features=embedding_size, out_features=num_class)    # circle loss + softplus
        loss = nn.Softplus()

    if osp.isfile(backbone_resume_root):
        if gpu is 0:
            logger.info("Loading Backbone Checkpoint '{}'".format(backbone_resume_root))
        load_pretrained_weights(backbone, backbone_resume_root, map_location="cpu")
    else:
        if gpu is 0:
            logger.info("No Checkpoint Found at '{}'. Train from Scratch".format(backbone_resume_root))

    if osp.isfile(head_resume_root):
        if gpu is 0:
            logger.info("Loading Head Checkpoint '{}'".format(head_resume_root))
        load_pretrained_weights(head, head_resume_root, map_location="cpu")
    else:
        if gpu is 0:
            logger.info("No Checkpoint Found at '{}'. Train from Scratch".format(head_resume_root))

    if "IR" in backbone_name:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_ir_bn_paras(backbone)
        _, head_paras_wo_bn = separate_ir_bn_paras(head)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone)
        _, head_paras_wo_bn = separate_resnet_bn_paras(head)

    # optimizer = optim.Adam([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': weight_decay},
    #                        {'params': backbone_paras_only_bn}], lr=LR)  # , momentum=momentum)

    # optimizer = optim.SGD([{'params': backbone_paras_wo_bn + list(head.parameters()), 'weight_decay': weight_decay},
    #                        {'params': backbone_paras_only_bn}], lr=LR, momentum=momentum)

    optimizer = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': weight_decay},
                           {'params': backbone_paras_only_bn}], lr=LR, momentum=momentum)

    backbone.cuda(gpu)
    head.cuda(gpu)
    ######################################################################
    # Wrap the model
    backbone = nn.parallel.DistributedDataParallel(backbone, device_ids=[gpu])
    head = nn.parallel.DistributedDataParallel(head, device_ids=[gpu])
    ######################################################################

    # ======= train & validation & save checkpoint =======#
    disp_batch = len(train_loader) // 10  # frequency to display training loss & acc

    warm_up_batch = len(train_loader) * warm_up_epoch

    batch = 0  # batch index

    loss_record = AverageMeter()
    top1_record = AverageMeter()
    top5_record = AverageMeter()

    for epoch in range(continue_epoch, num_epoch):
        if epoch in stages:
            schedule_lr(optimizer, rate=stages[epoch])

        if epoch == 0 and open_backbone_epoch != 0:
            close_all_layers(backbone)  # close backbone all layer
            if gpu is 0:
                logger.info("close all backbone layer")

        if epoch == open_backbone_epoch and open_backbone_epoch != 0:
            open_all_layers(backbone)   # open backbone all layer
            if gpu is 0:
                logger.info("open all backbone layer")

        backbone.train()  # set to training mode
        head.train()

        if epoch < continue_epoch:
            continue

        if gpu is 0:
            data_iter = tqdm(iter(train_loader))
        else:
            data_iter = train_loader

        # for inputs, labels in data_iter:
        for inputs, labels, yaw in data_iter:
            # adjust LR for each training batch during warm up
            if (epoch + 1 <= warm_up_epoch) and (batch + 1 <= warm_up_batch):
                warm_up_lr(batch + 1, warm_up_batch, LR, optimizer)

            # compute output
            inputs = inputs.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True).long()
            yaw = yaw.cuda(gpu,non_blocking=True).float()
            # features = backbone(inputs)
            features = backbone(inputs,yaw)
            outputs = head(features, labels)

            if head_name != 'CircleLoss':
                losses = loss(outputs, labels)
            else:
                losses = loss(outputs).mean()

            # measure accuracy and record loss
            if gpu is 0:
                if head_name != 'CircleLoss':
                    prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                else:
                    prec1, prec5 = accuracy(features.data, labels, topk=(1, 5))

                loss_record.update(losses.data.item(), inputs.size(0))
                top1_record.update(prec1.data.item(), inputs.size(0))
                top5_record.update(prec5.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # dispaly training loss & acc every disp_batch
            if gpu is 0 and ((batch + 1) % disp_batch == 0) and batch != 0:
                logger.info("=" * 60)
                logger.info(f"Epoch: {epoch + 1}/{num_epoch} Batch {batch + 1}/{len(train_loader) * num_epoch}\t"
                            f"Loss {loss_record.val:.3f} (mean:{loss_record.avg:.3f})\t"
                            f"Prec@1 {top1_record.val:.3f} (mean:{top1_record.avg:.3f})\t"
                            f"Prec@5 {top5_record.val:.3f} (mean:{top5_record.avg:.3f})\t"
                            f"lr {optimizer.param_groups[0]['lr']:.2e}")
                logger.info("=" * 60)

            batch += 1  # batch index

        # training statistics per epoch (buffer for visualization)
        if gpu is 0:
            # save checkpoints per epoch
            date_time = get_time()
            torch.save(backbone.module.state_dict(),
                       os.path.join(model_root,
                                    f"{date_time}_{backbone_name}_Epoch_{epoch + 1}_LOSS_{loss_record.avg:.3f}.pth"))
            torch.save(head.module.state_dict(),
                       os.path.join(model_root,
                                    f"{date_time}_{head_name}_Epoch_{epoch + 1}_LOSS_{loss_record.avg:.3f}.pth"))

            with SummaryWriter(log_root) as writer:
                writer.add_scalar("Training_Loss", loss_record.avg, epoch + 1)
                writer.add_scalar("Training_Accuracy", top1_record.avg, epoch + 1)

            logger.info("=" * 60)
            logger.info(f'Epoch: {epoch + 1}/{num_epoch}\t'
                        f'Training Loss {loss_record.val:.4f} ({loss_record.avg:.4f})\t'
                        f'Training Prec@1 {top1_record.val:.3f} ({top1_record.avg:.3f})\t'
                        f'Training Prec@5 {top5_record.val:.3f} ({top5_record.avg:.3f})')
            logger.info("=" * 60)

            # clean record prepare for next epoch
            loss_record.clean()
            top1_record.clean()
            top5_record.clean()


        # perform validation & save checkpoints per epoch
        # validation statistics per epoch (buffer for visualization)
        if epoch % eval_epoch == eval_epoch-1:
            if gpu is 0:
                acc_lfw, best_threshold_lfw, roc_curve_lfw = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                         embedding_size, batch_size,
                                                                         backbone, lfw, lfw_isSame, yaw)
                with SummaryWriter(log_root) as writer:
                    writer.add_scalar('{}_Accuracy'.format("LFW"), acc_lfw, epoch+1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: LFW: {acc_lfw:.5f}"
                            f" Threshold: {best_threshold_lfw:.3f}")
                logger.info("=" * 60)

            if gpu is 1:
                acc_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                                  embedding_size, batch_size,
                                                                                  backbone, cfp_ff, cfp_ff_isSame, yaw)
                with SummaryWriter(log_root) as writer:
                    writer.add_scalar('{}_Accuracy'.format("CFP_FF"), acc_cfp_ff, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CFP_FF: {acc_cfp_ff:.5f}"
                            f" Threshold: {best_threshold_cfp_ff:.3f}")
                logger.info("=" * 60)

            if gpu is 2:
                acc_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                                  embedding_size, batch_size,
                                                                                  backbone, cfp_fp, cfp_fp_isSame, yaw)
                with SummaryWriter(log_root) as writer:
                    writer.add_scalar('{}_Accuracy'.format("CFP_FP"), acc_cfp_fp, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CFP_FP: {acc_cfp_fp:.5f}"
                            f" Threshold: {best_threshold_cfp_fp:.3f}")
                logger.info("=" * 60)

            if gpu is 3:
                acc_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                               embedding_size, batch_size,
                                                                               backbone, cplfw, cplfw_isSame, yaw)
                with SummaryWriter(log_root) as writer:
                    writer.add_scalar('{}_Accuracy'.format("CPLFW"), acc_cplfw, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CPLFW: {acc_cplfw:.5f}"
                            f" Threshold: {best_threshold_cplfw:.3f}")
                logger.info("=" * 60)

            if gpu is 4:
                acc_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                                     embedding_size, batch_size,
                                                                                     backbone, vgg2_fp, vgg2_fp_isSame, yaw)
                with SummaryWriter(log_root) as writer:
                    writer.add_scalar('{}_Accuracy'.format("VGGFace2_FP"), acc_vgg2_fp, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: VGG2_FP: {acc_vgg2_fp:.5f}"
                            f" Threshold: {best_threshold_vgg2_fp:.3f}")
                logger.info("=" * 60)

            if gpu is 5:
                acc_agedb, best_threshold_agedb, roc_curve_agedb = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                               embedding_size, batch_size,
                                                                               backbone, agedb, agedb_isSame, yaw)
                with SummaryWriter(log_root) as writer:
                    writer.add_scalar('{}_Accuracy'.format("AgeDB"), acc_agedb, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: AgeDB: {acc_agedb:.5f}"
                            f" Threshold: {best_threshold_agedb:.3f}")
                logger.info("=" * 60)

            if gpu is 6:
                acc_calfw, best_threshold_calfw, roc_curve_calfw = perform_val_yaw(False, torch.device(f"cuda:{gpu}"),
                                                                               embedding_size, batch_size,
                                                                               backbone, calfw, calfw_isSame, yaw)
                with SummaryWriter(log_root) as writer:
                    writer.add_scalar('{}_Accuracy'.format("CALFW"), acc_calfw, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CALFW: {acc_calfw:.5f}"
                            f" Threshold: {best_threshold_calfw:.3f}")
                logger.info("=" * 60)


        # if epoch % eval_epoch == eval_epoch-1:
        #     if gpu is 0:
        #         acc_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(False, torch.device(f"cuda:{gpu}"),
        #                                                                  embedding_size, batch_size,
        #                                                                  backbone, lfw, lfw_issame)
        #         with SummaryWriter(log_root) as writer:
        #             writer.add_scalar('{}_Accuracy'.format("LFW"), acc_lfw, epoch+1)
        #         logger.info("=" * 60)
        #         logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: LFW: {acc_lfw:.5f}"
        #                     f" Threshold: {best_threshold_lfw:.3f}")
        #         logger.info("=" * 60)
        #
        #     if gpu is 1:
        #         acc_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(False, torch.device(f"cuda:{gpu}"),
        #                                                                           embedding_size, batch_size,
        #                                                                           backbone, cfp_ff, cfp_ff_issame)
        #         with SummaryWriter(log_root) as writer:
        #             writer.add_scalar('{}_Accuracy'.format("CFP_FF"), acc_cfp_ff, epoch + 1)
        #         logger.info("=" * 60)
        #         logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CFP_FF: {acc_cfp_ff:.5f}"
        #                     f" Threshold: {best_threshold_cfp_ff:.3f}")
        #         logger.info("=" * 60)
        #
        #     if gpu is 2:
        #         acc_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(False, torch.device(f"cuda:{gpu}"),
        #                                                                           embedding_size, batch_size,
        #                                                                           backbone, cfp_fp, cfp_fp_issame)
        #         with SummaryWriter(log_root) as writer:
        #             writer.add_scalar('{}_Accuracy'.format("CFP_FP"), acc_cfp_fp, epoch + 1)
        #         logger.info("=" * 60)
        #         logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CFP_FP: {acc_cfp_fp:.5f}"
        #                     f" Threshold: {best_threshold_cfp_fp:.3f}")
        #         logger.info("=" * 60)
        #
        #     if gpu is 3:
        #         acc_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(False, torch.device(f"cuda:{gpu}"),
        #                                                                        embedding_size, batch_size,
        #                                                                        backbone, cplfw, cplfw_issame)
        #         with SummaryWriter(log_root) as writer:
        #             writer.add_scalar('{}_Accuracy'.format("CPLFW"), acc_cplfw, epoch + 1)
        #         logger.info("=" * 60)
        #         logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CPLFW: {acc_cplfw:.5f}"
        #                     f" Threshold: {best_threshold_cplfw:.3f}")
        #         logger.info("=" * 60)
        #
        #     if gpu is 4:
        #         acc_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(False, torch.device(f"cuda:{gpu}"),
        #                                                                              embedding_size, batch_size,
        #                                                                              backbone, vgg2_fp, vgg2_fp_issame)
        #         with SummaryWriter(log_root) as writer:
        #             writer.add_scalar('{}_Accuracy'.format("VGGFace2_FP"), acc_vgg2_fp, epoch + 1)
        #         logger.info("=" * 60)
        #         logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: VGG2_FP: {acc_vgg2_fp:.5f}"
        #                     f" Threshold: {best_threshold_vgg2_fp:.3f}")
        #         logger.info("=" * 60)
        #
        #     if gpu is 5:
        #         acc_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(False, torch.device(f"cuda:{gpu}"),
        #                                                                        embedding_size, batch_size,
        #                                                                        backbone, agedb, agedb_issame)
        #         with SummaryWriter(log_root) as writer:
        #             writer.add_scalar('{}_Accuracy'.format("AgeDB"), acc_agedb, epoch + 1)
        #         logger.info("=" * 60)
        #         logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: AgeDB: {acc_agedb:.5f}"
        #                     f" Threshold: {best_threshold_agedb:.3f}")
        #         logger.info("=" * 60)
        #
        #     if gpu is 6:
        #         acc_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(False, torch.device(f"cuda:{gpu}"),
        #                                                                        embedding_size, batch_size,
        #                                                                        backbone, calfw, calfw_issame)
        #         with SummaryWriter(log_root) as writer:
        #             writer.add_scalar('{}_Accuracy'.format("CALFW"), acc_calfw, epoch + 1)
        #         logger.info("=" * 60)
        #         logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CALFW: {acc_calfw:.5f}"
        #                     f" Threshold: {best_threshold_calfw:.3f}")
        #         logger.info("=" * 60)




if __name__ == '__main__':
    main()