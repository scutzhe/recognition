#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : analysispth.py
# @time    : 7/13/20 4:17 PM
# @desc    : 
'''
import torch
modlePath = "model/model_best.pth.tar"
net = torch.load(modlePath)
print("type(net)=",type(net)) # class dict
print("len(net)=",len(net)) # 5
print("net.keys()=",net.keys()) # dict_keys(['epoch', 'arch', 'state_dict', 'best_prec1', 'optimizer'])

# print('net["state_dict"]=',net["state_dict"])
print("net['epoch']=",net['epoch'])
print("net['arch']=",net['arch'])
print("net['best_prec1']=",net['best_prec1'])
# print("net['optimizer']=",net['optimizer'])

