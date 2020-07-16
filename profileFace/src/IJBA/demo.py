#!/usr/bin/env python
# encoding: utf-8
'''
@author: 郑祥忠 
@license: (C) Copyright 2013-2019, 海格星航
@contact: dylenzheng@gmail.com 
@project: DREAM
@file: demo.py
@time: 8/28/19 7:09 PM
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
# import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from selfDefine import CFPDataset, CaffeCrop

from ResNet import resnet18, resnet50, resnet101

from test_recog import test_recog
from test_verify import test_verify

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir', metavar='DIR', default='/home/zhex/test_result/recognition/', help='path to dataset')
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
parser.add_argument('-b', '--batch-size', default=128, type=int,
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



def extract_feat(arch, model_path):
    global args, best_prec1
    args = parser.parse_args()

    if arch.find('end2end') >= 0:
        end2end = True
    else:
        end2end = False

    arch = arch.split('_')[0]

    class_num = 87020
    # class_num = 13386

    # create net
    model = None
    assert (arch in ['resnet18', 'resnet50', 'resnet101','resnet152'])
    if arch == 'resnet18':
        model = resnet18(pretrained=False, num_classes=class_num, \
                         extract_feature=True, end2end=end2end)
    if arch == 'resnet50':
        model = resnet50(pretrained=False, num_classes=class_num, \
                         extract_feature=True, end2end=end2end)
    if arch == 'resnet101':
        model = resnet101(pretrained=False, num_classes=class_num, \
                          extract_feature=True, end2end=end2end)
    if arch == 'resnet152':
        model = resnet152(pretrained=False, num_classes=class_num, \
                          extract_feature=True, end2end=end2end)

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # load model
    assert (os.path.isfile(model_path))
    checkpoint = torch.load(model_path)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
    for key in pretrained_state_dict:
        if key in model_state_dict:
            model_state_dict[key] = pretrained_state_dict[key]
    model.load_state_dict(model_state_dict)
    print('load trained model complete')

    caffe_crop = CaffeCrop('test')

    infos = [('/home/zhex/test_result/recognition','gallery_probe_dream','gallery'),
             ('/home/zhex/test_result/recognition','gallery_probe_dream','probe')]

    for root_dir, sub_dir, img_type in infos:
        split_dir = os.path.join(root_dir, sub_dir)
        img_dir = os.path.join(split_dir, img_type)

        img_list_file = os.path.join(split_dir, '{}.txt'.format(img_type))

        # preprocess dataset
        img_dataset = CFPDataset(args.img_dir, img_list_file,
            transforms.Compose([caffe_crop, transforms.ToTensor()]))

        # read dataset
        img_loader = torch.utils.data.DataLoader(
            img_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        for input in img_loader: #shape(input) = (128,3,224,224)
            # yaw = yaw.float().cuda(async=True)
            # yaw_var = torch.autograd.Variable(yaw)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
            # output = model(input_var, yaw_var)
            output = model(input_var)
            output_data = output.cpu().data.numpy()
            feat_num = output.size(0) # 128

        print('we have complete {}'.format(img_type))

def main():
    infos = [# ['resnet18_naive', '/home/zhex/pre_models/dream/ijba_res18_naive.pth.tar','nonli']
             ['resnet50_naive', '/home/zhex/pre_models/dream/cfp_res50_naive.pth.tar']]
    for arch, model_path in infos:
        print(model_path.split('/')[-1])
        extract_feat(arch, model_path)
        arch = arch.split('_')[0]
        print('arch=',arch)
        print('model_path=',model_path)
        # test_recog(arch)
        # test_verify(arch)
main()