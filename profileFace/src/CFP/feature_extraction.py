#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2013-2019,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : feature_extraction.py
# @time    : 12/17/19 11:00 AM
# @desc    : 
'''

import argparse
import os, sys, shutil
import time
import struct as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from selfDefine import CFPDataset, CaffeCrop
from ResNet import resnet18, resnet50, resnet101
from eval_roc import eval_roc_main

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir', metavar='DIR', default='', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='/home/zhex/pre_models/dream/cfp_res50_end2end.pth.tar', type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--model_dir', '-m', default='./model', type=str)
args = parser.parse_args()

class_num = 13386
# create net
model = resnet50(pretrained=False, num_classes=class_num,extract_feature=True, end2end=True)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(args.resume)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

caffe_crop = CaffeCrop('test')
transform = transforms.Compose([caffe_crop,transforms.ToTensor()])


def extract_feat(img):
    '''
    :param img: img_data
    :return: 1*256 numpy_array
    '''
    img_tensor = transform(img)
    # print('img_shape=',img_tensor.size())
    img_tensor = img_tensor.unsqueeze(0)
    # print('img_tensor.size=',img_tensor.size())
    output = model(img_tensor)
    return output

def cal_distance(feature1,feature2):
    '''
    :param img1: img_data
    :param img2: img_data
    :return: similarity num_value
    '''
    similarity = torch.nn.CosineSimilarity()(feature1, feature2)
    sim = similarity.cpu().detach().numpy()[0]
    return sim

def main():
    img_dir1 = '/home/zhex/test_result/profile_face/chenwenhao/01'
    img_dir2 = '/home/zhex/test_result/profile_face/chenwenhao/02'
    img_names1 = os.listdir(img_dir1)
    img_names2 = os.listdir(img_dir2)
    length1 = len(img_names1)
    img_names1.sort()

    length2 = len(img_names2)
    img_names2.sort()

    for i in range(length1):
        img_name1 = img_names1[i]
        img_path1 = os.path.join(img_dir1, img_name1)
        img1 = Image.open(img_path1).convert('RGB')

        for j in range(length2):
            img_name2 = img_names2[j]
            img_path2 = os.path.join(img_dir2, img_name2)
            img2 = Image.open(img_path2).convert('RGB')

            features1 = extract_feat(img1)
            features2 = extract_feat(img2)

            similarity = cal_distance(features1, features2)

            print("{} and {}'s distance is = ".format(img_path1.split('/', 6)[6], img_path2.split('/', 6)[6]),
                  ("%.4f" % similarity))

main()