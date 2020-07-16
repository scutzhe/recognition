#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : splitDataset.py
# @time    : 7/16/20 2:12 PM
# @desc    : 
'''

import os

def splitDateset(imgDir,originImagePath,originLabelPath,trainImagePath,trainLabelPath,valImagePath,valLabelPath):
    """
    :param imgDir:
    :param originImagePath:
    :param originLabelPath:
    :param trainImagePath:
    :param trainLabelPath:
    :param valImagePath:
    :param valLabelPath:
    :return:
    """
    assert os.path.exists(imgDir),"{} is null !!!".format(imgDir)
    assert os.path.exists(originImagePath),"{} is null !!!".format(originImagePath)
    assert os.path.exists(originLabelPath),"{} is null !!!".format(originLabelPath)

    valID = []
    tmpIndex = 0
    for id in os.listdir(imgDir):
        tmpIndex += 1
        valID.append(id)
        if tmpIndex == 9400:
            break
    print("len(valID)=",len(valID))
    imageIndex = 0
    labelIndex = 0
    trainImageIndex = 0
    trainLabelIndex = 0
    valImageIndex = 0
    valLabelIndex = 0

    originImageFile = open(originImagePath,"r")
    originLabelFile = open(originLabelPath,"r")
    trainImageFile = open(trainImagePath,"a")
    trainLabelFile = open(trainLabelPath,"a")
    valImageFile = open(valImagePath,"a")
    valLabelFile = open(valLabelPath,"a")

    for lineImage in originImageFile.readlines():
        imageIndex += 1
        if lineImage.strip() in valID:
            valImageFile.write(lineImage)
            valImageIndex += 1
        else:
            trainImageFile.write(lineImage)
            trainImageIndex += 1

    for lineLabel in originLabelFile.readlines():
        labelIndex += 1
        if lineLabel.strip().split("/")[0] in valID:
            valLabelFile.write(lineLabel)
            valLabelIndex += 1
        else:
            trainLabelFile.write(lineLabel)
            trainLabelIndex +=1

    print("imageIndex=",imageIndex)
    print("labelIndex=",labelIndex)
    print("trainImageIndex=",trainImageIndex)
    print("valImageIndex=",valImageIndex)
    print("trainLabelIndex=",trainLabelIndex)
    print("valLabelIndex=",valLabelIndex)

if __name__  == "__main__":
    imgDir = "/home/zhex/data/imgs_glintasia"
    originImagePath = "annotation/originImage.txt"
    originLabelPath = "annotation/originLabel.txt"
    trainImagePath = "annotation/trainImage.txt"
    trainLabelPath = "annotation/trainLabel.txt"
    valImagePath = "annotation/valImage.txt"
    valLabelPath = "annotation/valLabel.txt"
    splitDateset(imgDir,originImagePath,originLabelPath,trainImagePath,trainLabelPath,valImagePath,valLabelPath)