#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : compare_index.py
# @time    : 8/13/20 4:26 PM
# @desc    : 
'''

import os
import cv2
from tqdm import tqdm
from face_classification.classification import FaceClass
faceClasser = FaceClass(model_path='/home/zhex/pre_models/AILuoGang/faceclassification.trt')


if __name__ == "__main__":
    frontalFaceDir = "/home/zhex/test_result/faceYaw/luogangjiankongtest/frontal"
    profileFaceDir = "/home/zhex/test_result/faceYaw/luogangjiankongtest/profile"
    frontalNames = os.listdir(frontalFaceDir)
    profileNames = os.listdir(profileFaceDir)
    totalNumFrontal = len(frontalNames)
    totalNumProfile = len(profileNames)
    numFrontal = 0
    numProfile = 0
    numF = 0
    numP = 0
    num = 0
    total = totalNumFrontal + totalNumProfile
    for imageName in tqdm(frontalNames):
        imagePath = os.path.join(frontalFaceDir,imageName)
        image = cv2.imread(imagePath)
        h,w = image.shape[:2]
        image = image[h//4:3*h//4,w//4:3*w//4,:]
        flag = min(image.shape[0],image.shape[1])
        if flag < 40:
            numF += 1
            continue
        imageBGR = cv2.resize(image,(112,112),interpolation=cv2.INTER_CUBIC)
        imageRGB = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB)
        label = faceClasser.classifyNormal([imageRGB])
        if label == 0:
            numFrontal += 1

    for imageName in tqdm(profileNames):
        imagePath = os.path.join(profileFaceDir,imageName)
        image = cv2.imread(imagePath)
        h, w = image.shape[:2]
        image = image[h // 4:3 * h // 4, w // 4:3 * w // 4, :]
        flag = min(image.shape[0], image.shape[1])
        if flag < 40:
            numP += 1
            continue
        imageBGR = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)
        imageRGB = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB)
        label = faceClasser.classifyNormal([imageRGB])
        if label == 1:
            numProfile += 1

    num = numFrontal + numProfile
    # print("numFrontal=",numFrontal)
    # print("totalNumFrontal=",totalNumFrontal)
    # print("numProfile=",numProfile)
    # print("totalNumProfile=",totalNumProfile)
    print("origin algorithm")
    print("numFrontal/(totalNumFrontal - numF)=",numFrontal/(totalNumFrontal - numF))
    print("numProfile/(totalNumProfile - numP)=",numProfile/(totalNumProfile - numP))
    print("num/(total - numF -numP)=",round(num/(total - numF - numP),4))
