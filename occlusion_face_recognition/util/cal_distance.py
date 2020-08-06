#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : cal_distance.py
# @time    : 8/6/20 2:25 PM
# @desc    : 
'''

import torch
def cal_distance(feature1,feature2):
    '''
    :param img1: img_data
    :param img2: img_data
    :return: similarity num_value
    '''
    similarity = torch.nn.CosineSimilarity()(feature1, feature2)
    distance = similarity.cpu().detach().numpy()[0]
    return distance