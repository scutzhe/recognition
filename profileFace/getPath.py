#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : getPath.py
# @time    : 7/9/20 9:27 PM
# @desc    : 
'''

import os
rootPath = "/home/shengyang/haige_dataset/face_occusion/imgs_glintasia"
trainList = open("trainList.txt","a")
num = 0
for root, dirs, names in os.walk(rootPath):
    for imgName in names:
        imgPath = os.path.join(root,imgName)
        trainList.write(imgPath + "\n")
        num += 1
        print("num=",num)
print("num=",num)