#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: train_ori.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.10.21 16:09    samon      v0.1        creation
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# from opencv_transforms import transforms
import torchvision.datasets as datasets

from config_ori_ddp import configurations
from backbone.model_irse import IR_SE_50, IR_SE_101
from backbone.efficient_face_model import EfficientFaceModel
from head.metrics import ArcFace
from loss.focal import FocalLoss
# from util.utils import make_weights_for_balanced_classes, get_val_data, buffer_val
from util.utils import get_val_pair, separate_ir_bn_paras, separate_resnet_bn_paras
from util.utils import warm_up_lr, schedule_lr, perform_val, get_time, AverageMeter, accuracy

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import argparse
import torch.multiprocessing as mp
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

import torch.distributed as dist
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(asctime)s]-[pid:%(process)d]-%(message)s")

logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-g', '--gpus', default=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
                        type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    args = parser.parse_args()

    #########################################################
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '192.168.2.238'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    #########################################################


def train(gpu, args):
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

    data_root = cfg['DATA_ROOT']  # the parent root where your train/val/test data are stored
    model_root = cfg['MODEL_ROOT']
    log_root = cfg['LOG_ROOT']  # the root to log your train/val status

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
    weight_decay = cfg['WEIGHT_DECAY']
    momentum = cfg['MOMENTUM']
    stages = cfg['STAGES']  # epoch stages to decay learning rate
    num_workers = cfg['NUM_WORKERS']  # epoch stages to decay learning rate

    writer = SummaryWriter(log_root)  # writer for buffering intermedium results

    train_transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
        transforms.RandomCrop([input_size[0], input_size[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std),
    ])

    dataset_train = datasets.ImageFolder(os.path.join(data_root, 'imgs'), train_transform)

    ######################################################################
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                              num_replicas=args.world_size,
                                                              rank=rank)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                               sampler=sampler, pin_memory=True, shuffle=False,
                                               num_workers=num_workers, drop_last=drop_last)
    ######################################################################

    num_class = len(train_loader.dataset.classes)
    if gpu is 0:
        logger.info("Number of Training Classes: {}".format(num_class))

    lfw, lfw_issame = get_val_pair(data_root, 'lfw')
    cfp_ff, cfp_ff_issame = get_val_pair(data_root, 'cfp_ff')
    cfp_fp, cfp_fp_issame = get_val_pair(data_root, 'cfp_fp')
    agedb, agedb_issame = get_val_pair(data_root, 'agedb_30')
    calfw, calfw_issame = get_val_pair(data_root, 'calfw')
    cplfw, cplfw_issame = get_val_pair(data_root, 'cplfw')
    vgg2_fp, vgg2_fp_issame = get_val_pair(data_root, 'vgg2_fp')

    # ======= model & loss & optimizer =======#
    backbone = IR_SE_50(input_size)
    # backbone = EfficientFaceModel.from_name(backbone_name)

    head = ArcFace(in_features=embedding_size, out_features=num_class, device_id=[gpu])
    LOSS = FocalLoss()

    if backbone_name.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_ir_bn_paras(backbone)
        _, head_paras_wo_bn = separate_ir_bn_paras(head)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone)
        _, head_paras_wo_bn = separate_resnet_bn_paras(head)
    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': weight_decay},
                           {'params': backbone_paras_only_bn}], lr=LR, momentum=momentum)

    backbone.cuda(gpu)
    head.cuda(gpu)
    ######################################################################
    # Wrap the model
    [backbone, head], OPTIMIZER = amp.initialize([backbone, head], OPTIMIZER, opt_level='O2')
    backbone = DDP(backbone)
    head = DDP(head)
    ######################################################################

    # ======= train & validation & save checkpoint =======#
    DISP_FREQ = len(train_loader) // 10  # frequency to display training loss & acc

    NUM_EPOCH_WARM_UP = num_epoch // 25  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up

    batch = 0  # batch index

    for epoch in range(num_epoch):  # start training process
        if epoch == stages[0]:
            schedule_lr(OPTIMIZER)
        if epoch == stages[1]:
            schedule_lr(OPTIMIZER)
        if epoch == stages[2]:
            schedule_lr(OPTIMIZER)

        backbone.train()  # set to training mode
        head.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        if gpu is 0:
            data_iter = tqdm(iter(train_loader))
        else:
            data_iter = train_loader

        for inputs, labels in data_iter:
            # adjust LR for each training batch during warm up
            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP):
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

            # compute output
            inputs = inputs.cuda()
            labels = labels.cuda().long()
            features = backbone(inputs)
            outputs = head(features, labels)
            loss = LOSS(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            with amp.scale_loss(loss, OPTIMIZER) as scaled_loss:
                scaled_loss.backward()

            OPTIMIZER.step()

            # dispaly training loss & acc every DISP_FREQ
            if gpu is 0 and ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                logger.info("=" * 60)
                logger.info(f"Epoch: {epoch + 1}/{num_epoch} Batch {batch + 1}/{len(train_loader) * num_epoch}\t"
                             f"Training Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                             f"Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                             f"Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t"
                             f"lr {OPTIMIZER.param_groups[0]['lr']:.5f}")
                logger.info("=" * 60)

            batch += 1  # batch index

        # training statistics per epoch (buffer for visualization)
        if gpu is 0:
            # save checkpoints per epoch
            torch.save(backbone.module.state_dict(),
                       os.path.join(model_root, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                           backbone_name, epoch + 1, batch, get_time())))
            torch.save(head.state_dict(),
                       os.path.join(model_root, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                           head_name, epoch + 1, batch, get_time())))

            epoch_loss = losses.avg
            epoch_acc = top1.avg
            writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
            writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
            logger.info("=" * 60)
            logger.info(f'Epoch: {epoch + 1}/{num_epoch}\t'
                         f'Training Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                         f'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         f'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})')
            logger.info("=" * 60)

        # perform validation & save checkpoints per epoch
        # validation statistics per epoch (buffer for visualization)
        if True:
            if gpu is 0:
                acc_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(False, torch.device(f"cuda:{gpu}"),
                                                                         embedding_size, batch_size,
                                                                         backbone, lfw, lfw_issame)
                # buffer_val(writer, "LFW", acc_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: LFW: {acc_lfw:.5f}"
                             f" Threshold: {best_threshold_lfw:.3f}")
                logger.info("=" * 60)

            if gpu is 1:
                acc_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(False, torch.device(f"cuda:{gpu}"),
                                                                                  embedding_size, batch_size,
                                                                                  backbone, cfp_ff, cfp_ff_issame)
                # buffer_val(writer, "CFP_FF", acc_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CFP_FF: {acc_cfp_ff:.5f}"
                             f" Threshold: {best_threshold_cfp_ff:.3f}")
                logger.info("=" * 60)

            if gpu is 2:
                acc_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(False, torch.device(f"cuda:{gpu}"),
                                                                                  embedding_size, batch_size,
                                                                                  backbone, cfp_fp, cfp_fp_issame)
                # buffer_val(writer, "CFP_FP", acc_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CFP_FP: {acc_cfp_fp:.5f}"
                             f" Threshold: {best_threshold_cfp_fp:.3f}")
                logger.info("=" * 60)

            if gpu is 3:
                acc_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(False, torch.device(f"cuda:{gpu}"),
                                                                               embedding_size, batch_size,
                                                                               backbone, agedb, agedb_issame)
                # buffer_val(writer, "AgeDB", acc_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: AgeDB: {acc_agedb:.5f}"
                             f" Threshold: {best_threshold_agedb:.3f}")
                logger.info("=" * 60)

            if gpu is 4:
                acc_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(False, torch.device(f"cuda:{gpu}"),
                                                                               embedding_size, batch_size,
                                                                               backbone, calfw, calfw_issame)
                # buffer_val(writer, "CALFW", acc_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CALFW: {acc_calfw:.5f}"
                             f" Threshold: {best_threshold_calfw:.3f}")
                logger.info("=" * 60)

            if gpu is 5:
                acc_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(False, torch.device(f"cuda:{gpu}"),
                                                                               embedding_size, batch_size,
                                                                               backbone, cplfw, cplfw_issame)
                # buffer_val(writer, "CPLFW", acc_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: CPLFW: {acc_cplfw:.5f}"
                             f" Threshold: {best_threshold_cplfw:.3f}")
                logger.info("=" * 60)

            if gpu is 6:
                acc_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(False, torch.device(f"cuda:{gpu}"),
                                                                                     embedding_size, batch_size,
                                                                                     backbone, vgg2_fp, vgg2_fp_issame)
                # buffer_val(writer, "VGGFace2_FP", acc_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
                logger.info("=" * 60)
                logger.info(f"Epoch {epoch + 1}/{num_epoch}, Evaluation Acc: VGG2_FP: {acc_vgg2_fp:.5f}"
                             f" Threshold: {best_threshold_vgg2_fp:.3f}")
                logger.info("=" * 60)


if __name__ == '__main__':
    main()
