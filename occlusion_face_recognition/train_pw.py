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


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
import torch.utils.data

from backbone.model_pw import PW_IR_SE_101
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_ir_bn_paras, get_val_pair
from util.utils import separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time
from util.utils import buffer_val, AverageMeter, accuracy

from config_pw import configurations
from loss.mask_loss import MaskLoss
from dataset.pw_data_folder import PWImageFolder
from util.checkpoint_tools import load_pretrained_weights, open_specified_layers, open_all_layers

from tensorboardX import SummaryWriter
from tqdm import tqdm


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

    writer = SummaryWriter(LOG_ROOT)  # writer for buffering intermedium results

    # dataset_train = PWImageFolder(image_dir=os.path.join(DATA_ROOT, 'subset_for_test', 'imgs'),
    #                               mask_dir=os.path.join(DATA_ROOT, 'subset_for_test', 'mask'),
    #                               input_size=INPUT_SIZE
    #                               )
    dataset_train = PWImageFolder(image_dir=os.path.join(DATA_ROOT, 'real_occlusion2', 'imgs'),
                                  mask_dir=os.path.join(DATA_ROOT, 'real_occlusion2', 'mask'),
                                  input_size=INPUT_SIZE
                                  )

    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(dataset_train.label, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS, drop_last=DROP_LAST
    )

    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))

    lfw, lfw_issame = get_val_pair(DATA_ROOT, 'lfw')

    with open(MASKSAMPLE_ROOT, 'rb') as f:
        mask_sample = torch.Tensor(pickle.load(f))

    BACKBONE = PW_IR_SE_101(INPUT_SIZE, masksample=mask_sample)
    # print("=" * 60)
    # print(BACKBONE)
    # print("=" * 60)

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

    # ======= train & validation & save checkpoint =======#
    DISP_FREQ = len(train_loader) // 100  # frequency to display training loss & acc

    # NUM_EPOCH_WARM_UP = NUM_EPOCH // 50  # use the first 1/50 epochs to warm up
    # NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/50 epochs to warm up
    NUM_EPOCH_WARM_UP = 0  # use the first 1/50 epochs to warm up
    NUM_BATCH_WARM_UP = 0  # use the first 1/50 epochs to warm up
    batch = 0  # batch index

    train_step = [0, 0]       # just train HEAD

    # open_all_layers(BACKBONE)  # open BACKBONE all layer
    # open_all_layers(HEAD)  # open HEAD all layer
    # BACKBONE.no_attention = False

    for epoch in range(NUM_EPOCH):  # start training process
        # adjust LR for each training stage after warm up,
        # you can also choose to adjust LR manually (with slight modification) once plaueau observed
        if epoch == STAGES[0]:
            schedule_lr(OPTIMIZER)
        elif epoch == STAGES[1]:
            schedule_lr(OPTIMIZER)
        elif epoch == STAGES[2]:
            schedule_lr(OPTIMIZER)
        if epoch == 0:
            open_specified_layers(BACKBONE, [])  # close backbone all layer
            open_all_layers(HEAD)     # open HEAD all layer
            BACKBONE.no_attention = True        # no attention
        if epoch == train_step[0]:
            open_specified_layers(BACKBONE, ['levelattention', ])   # open attention layer
            open_specified_layers(HEAD, [])   # close HEAD all layer
            BACKBONE.no_attention = False
        if epoch == train_step[1]:
            open_all_layers(BACKBONE)   # open BACKBONE all layer
            open_all_layers(HEAD)       # open HEAD all layer
            BACKBONE.no_attention = False

        BACKBONE.train()  # set to training mode
        HEAD.train()

        identify_losses = AverageMeter()
        feature_losses = AverageMeter()
        mask_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # for inputs, masks_gt, labels in tqdm(iter(train_loader)):
        for occ_sample, occ_mask_gt, clean_sample, clean_mask_gt, label in tqdm(iter(train_loader)):
            loaded_batch = occ_sample.size(0)
            # adjust LR for each training batch during warm up
            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP):
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

            occ_mask_gt = occ_mask_gt.cuda()
            clean_mask_gt = clean_mask_gt.cuda()
            occ_sample = occ_sample.to(DEVICE)
            clean_sample = clean_sample.to(DEVICE)
            label = label.to(DEVICE).long()

            occ_mask, occ_feature, clean_mask, clean_feature = BACKBONE([occ_sample, clean_sample])
            if epoch < train_step[0]:     # just backward identify loss
                occ_output = HEAD(occ_feature, label)
                clean_output = HEAD(clean_feature, label)
                outputs = torch.cat([occ_output, clean_output], dim=0)
                labels = torch.cat([label, label], dim=0)

                occ_recognition_loss = LOSS(occ_output, label)
                clean_recognition_loss = LOSS(clean_output, label)
                recognition_loss = (occ_recognition_loss + clean_recognition_loss) / 2

                occ_mask_loss = MASKLOSS(occ_mask, occ_mask_gt)
                clean_mask_loss = MASKLOSS(clean_mask, clean_mask_gt)
                mask_loss = (occ_mask_loss + clean_mask_loss) / 2

                feature_loss = FEATURELOSS(occ_feature, clean_feature, torch.ones(loaded_batch, 1).cuda())

                loss = recognition_loss + feature_loss
                identify_losses.update(occ_recognition_loss.data.item(), loaded_batch)
                mask_losses.update(mask_loss.data.item(), loaded_batch)
                feature_losses.update(feature_loss.data.item(), loaded_batch)

            elif train_step[0] < epoch < train_step[1]:     # just backward attention loss
                occ_output = HEAD(occ_feature, label)
                clean_output = HEAD(clean_feature, label)
                outputs = torch.cat([occ_output, clean_output], dim=0)
                labels = torch.cat([label, label], dim=0)

                occ_recognition_loss = LOSS(occ_output, label)
                clean_recognition_loss = LOSS(clean_output, label)
                occ_mask_loss = MASKLOSS(occ_mask, occ_mask_gt)
                clean_mask_loss = MASKLOSS(clean_mask, clean_mask_gt)
                mask_loss = (occ_mask_loss + clean_mask_loss) / 2

                feature_loss = FEATURELOSS(occ_feature, clean_feature, torch.ones(loaded_batch, 1).cuda())

                loss = mask_loss + feature_loss
                identify_losses.update(occ_recognition_loss.data.item(), loaded_batch)
                mask_losses.update(mask_loss.data.item(), loaded_batch)
                feature_losses.update(feature_loss.data.item(), loaded_batch)
            else:       # backward all loss
                occ_output = HEAD(occ_feature, label)
                clean_output = HEAD(clean_feature, label)
                outputs = torch.cat([occ_output, clean_output], dim=0)
                labels = torch.cat([label, label], dim=0)

                occ_recognition_loss = LOSS(occ_output, label)
                clean_recognition_loss = LOSS(clean_output, label)
                recognition_loss = (occ_recognition_loss + clean_recognition_loss) / 2

                occ_mask_loss = MASKLOSS(occ_mask, occ_mask_gt)
                clean_mask_loss = MASKLOSS(clean_mask, clean_mask_gt)
                mask_loss = (occ_mask_loss + clean_mask_loss) / 2

                feature_loss = FEATURELOSS(occ_feature, clean_feature, torch.ones(loaded_batch, 1).cuda())

                loss = mask_loss + recognition_loss + feature_loss

                identify_losses.update(occ_recognition_loss.data.item(), loaded_batch)
                mask_losses.update(mask_loss.data.item(), loaded_batch)
                feature_losses.update(feature_loss.data.item(), loaded_batch)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            top1.update(prec1.data.item(), loaded_batch)
            top5.update(prec5.data.item(), loaded_batch)

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Identify Loss {id_loss.avg:.4f}\t'
                      'Mask Loss {mask_loss.avg:.4f}\t'
                      'Feature Loss {fea_loss.avg:.4f}\t'
                      'Top1 {top1.avg:.3f}\t'
                      'Top@5 {top5.avg:.3f}\t'
                      'lr {lr:.5f}'.format(
                    epoch + 1, NUM_EPOCH, batch + 1, len(train_loader) * NUM_EPOCH, id_loss=identify_losses,
                    mask_loss=mask_losses, fea_loss=feature_losses, top1=top1, top5=top5,
                    lr=OPTIMIZER.param_groups[0]['lr']))
                print("=" * 60)

            batch += 1  # batch index

        # training statistics per epoch (buffer for visualization)
        epoch_identify_loss = identify_losses.avg
        epoch_mask_loss = mask_losses.avg
        epoch_acc = top1.avg
        writer.add_scalar("Training_Identify_Loss", epoch_identify_loss, epoch + 1)
        writer.add_scalar("Training_Mask_Loss", epoch_mask_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Identify Loss {id_loss.avg:.4f}\t'
              'Mask Loss {mask_loss.avg:.4f}\t'
              'Feature Loss {fea_loss.avg:.4f}\t'
              'Training Prec@1 {top1.avg:.3f}\t'
              'Training Prec@5 {top5.avg:.3f}'.format(
            epoch + 1, NUM_EPOCH, id_loss=identify_losses, mask_loss=mask_losses,
            fea_loss=feature_losses, top1=top1, top5=top5))
        print("=" * 60)

        # perform validation & save checkpoints per epoch
        # validation statistics per epoch (buffer for visualization)
        print("=" * 60)
        print("Perform Evaluation on LFW and Save Checkpoints...")
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE,
                                                                      BACKBONE, lfw, lfw_issame)
        buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
        print("Epoch {}/{}, Evaluation: LFW Acc: {}".format(epoch + 1, NUM_EPOCH, accuracy_lfw))
        print("=" * 60)

        # accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
        #                                                                        BATCH_SIZE, BACKBONE, cfp_ff,
        #                                                                        cfp_ff_issame)
        # buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
        # accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
        #                                                                        BATCH_SIZE, BACKBONE, cfp_fp,
        #                                                                        cfp_fp_issame)
        # buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
        # accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
        #                                                                     BATCH_SIZE, BACKBONE, agedb, agedb_issame)
        # buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
        # accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
        #                                                                     BATCH_SIZE, BACKBONE, calfw, calfw_issame)
        # buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
        # accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
        #                                                                     BATCH_SIZE, BACKBONE, cplfw, cplfw_issame)
        # buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
        # accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
        #                                                                           BATCH_SIZE, BACKBONE, vgg2_fp,
        #                                                                           vgg2_fp_issame)
        # buffer_val(writer, "VGGFace2_FP", accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
        # print(
        #     "Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {}, CPLFW Acc: {}, VGG2_FP Acc: {}".format(
        #         epoch + 1, NUM_EPOCH, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, accuracy_calfw,
        #         accuracy_cplfw, accuracy_vgg2_fp))
        # print("=" * 60)

        # save checkpoints per epoch
        if MULTI_GPU:
            torch.save(BACKBONE.module.state_dict(), osp.join(MODEL_ROOT,
                                                                  "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                                      BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), osp.join(MODEL_ROOT,
                                                       "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                           HEAD_NAME, epoch + 1, batch, get_time())))
        else:
            torch.save(BACKBONE.state_dict(), osp.join(MODEL_ROOT,
                                                       "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                               BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), osp.join(MODEL_ROOT,
                                                       "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                           HEAD_NAME, epoch + 1, batch, get_time())))
