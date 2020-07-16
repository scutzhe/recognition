#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: demo.py.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.07.03 15:33    shengyang      v0.1        creation
from tensorboardX import SummaryWriter
import math
from datetime import datetime
import os
import os.path as osp
import time


# logs = ["IRSE50_ms1m+asia_lr0.02_circleloss_bs128x4",]

logs = ["IRSE101_ms1m+asia_lr0.02_arcface_bs128x4",]


for log in logs:
    log_dir = osp.expanduser(f"log/07-04_{log}")

    with open(f"log/{log}.log", "rt") as fp:
        all_lines = fp.readlines()

    acc_step = 0
    loss_step = 0
    lfw_acc_step = 0

    agedb_acc_step = 0
    cfpff_acc_step = 0
    cfpfp_acc_step = 0

    writer = SummaryWriter(log_dir=log_dir)

    for one_line in all_lines:
        if "Training Loss" in one_line:
            ind0 = one_line.find("(")
            ind1 = one_line.find(")")
            training_loss = float(one_line[ind0+1:ind1])

            sub_line = one_line[ind1+1:]
            ind0 = sub_line.find("(")
            ind1 = sub_line.find(")")
            training_acc = float(sub_line[ind0+1:ind1])

            loss_step += 1
            acc_step += 1
            # with SummaryWriter(log_dir=log_dir) as writer:
            writer.add_scalar("Training_Loss", training_loss, loss_step)
            writer.add_scalar("Training_Accuracy", training_acc, acc_step)
            print(training_loss, loss_step)

        elif "Evaluation Acc" in one_line:
            if "AgeDB" in one_line:
                ind0 = one_line.find("0.")
                ind1 = one_line.find(" Threshold")
                acc = float(one_line[ind0 + 1:ind1])
                agedb_acc_step += 5
                writer.add_scalar('{}_Accuracy'.format("AgeDB"), acc, agedb_acc_step)
            elif "LFW" in one_line:
                ind0 = one_line.find("0.")
                ind1 = one_line.find(" Threshold")
                acc = float(one_line[ind0 + 1:ind1])
                lfw_acc_step += 5
                writer.add_scalar('{}_Accuracy'.format("LFW"), acc, lfw_acc_step)
                # print(acc, lfw_acc_step)
            elif "CFP_FF" in one_line:
                ind0 = one_line.find("0.")
                ind1 = one_line.find(" Threshold")
                acc = float(one_line[ind0 + 1:ind1])
                cfpff_acc_step += 5
                writer.add_scalar('{}_Accuracy'.format("CFP_FF"), acc, cfpff_acc_step)
            elif "CFP_FP" in one_line:
                ind0 = one_line.find("0.")
                ind1 = one_line.find(" Threshold")
                acc = float(one_line[ind0 + 1:ind1])
                cfpfp_acc_step += 5
                writer.add_scalar('{}_Accuracy'.format("CFP_FP"), acc, cfpfp_acc_step)

    writer.close()