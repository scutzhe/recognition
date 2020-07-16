#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : settings.py
# @time    : 6/24/20 10:52 AM
# @desc    : 
'''
FACESIZE = 112 #输入网络的人脸图片尺寸
PIXSIZE = 40   #检测到人脸图片尺寸大于这个值才保留
FRAMESTEP = 3
FRAMESTEPMAX = 24
FACENUM = 30
FACEFACTOR = 2 #人脸放大倍数
PICNUM = 40
DIFFERENCE = 0.30
TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
ONEIDNUM = 10

symbol_file_path = 'faceDetectionLFFD/symbol_farm/symbol_10_320_20L_5scales_v2_deploy.json'
model_file_path = 'faceDetectionLFFD/saved_model/configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1000000.params'