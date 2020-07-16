#!/usr/bin/env python
# encoding: utf-8
'''
@author: 郑祥忠 
@license: (C) Copyright 2013-2019, 海格星航
@contact: dylenzheng@gmail.com 
@project: profileFace
@file: feature_extraction.py
@time: 9/15/19 9:27 AM
@desc:
'''
import argparse
import os, sys, shutil
import time
import struct as st

import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy

from src.CFP.selfDefine import CFPDataset
from src.end2end.selfDefine import CaffeCrop
from src.end2end.ResNet import resnet18, resnet50, resnet101, resnet152
from testIndex.testRecog import test_recog
from testIndex.testVerify import test_verify

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir', metavar='DIR', default='', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./data/resnet18.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--model_dir', '-m', default='', type=str)

args = parser.parse_args()


class_num = 87020

# create net
model = resnet18(pretrained=False, num_classes=class_num,extract_feature=True, end2end=True)
model = torch.nn.DataParallel(model).cuda()
model.eval()

# load model
print('start loading model ...')
model_path = '/home/zhex/pre_models/dream/ijba_res18_end2end.pth.tar'
assert (os.path.isfile(model_path))
checkpoint = torch.load(model_path)
pretrained_state_dict = checkpoint['state_dict']
model_state_dict = model.state_dict()
for key in pretrained_state_dict:
    if key in model_state_dict:
        model_state_dict[key] = pretrained_state_dict[key]
model.load_state_dict(model_state_dict)
print('load model successfully !!!')

caffe_crop = CaffeCrop('test')
transform = transforms.Compose([caffe_crop, transforms.ToTensor()])

def extract_feat(img):
    '''
    :param img: img_data
    :return: 1*256 numpy_array
    '''
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    output = model(img_tensor)
    return output

def cal_distance(feature1,feature2):
    '''
    :param img1: img_data
    :param img2: img_data
    :return: similarity num_value
    '''
    similarity = torch.nn.CosineSimilarity()(feature1, feature2)
    sim = similarity.cpu().detach().numpy()
    return sim

def main():
    # img_path1 = '/home/zhex/test_result/face_gallery_probe/gallery/02.png'
    # img_path2 = '/home/zhex/test_result/face_gallery_probe/gallery/06.png'
    img_path1 = '/home/zhex/work/face.evoLVe.PyTorch/align/hch/01.png'
    img_path2 = '/home/zhex/work/face.evoLVe.PyTorch/align/hch/08.png'
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)
    features1 = extract_feat(img1)
    features2 = extract_feat(img2)
    similarity = cal_distance(features1,features2)
    print("img1 and img2's distance is = ",similarity)

main()