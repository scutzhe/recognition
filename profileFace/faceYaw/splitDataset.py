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
from tqdm import tqdm
import shutil
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

    for lineImage in tqdm(originImageFile.readlines()):
        imageIndex += 1
        if lineImage.strip() in valID:
            valImageFile.write(lineImage)
            valImageIndex += 1
        else:
            trainImageFile.write(lineImage)
            trainImageIndex += 1

    for lineLabel in tqdm(originLabelFile.readlines()):
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

def splitDatesetNewID(imgDir,originImagePath,originLabelPath,trainImagePath,trainLabelPath,valImagePath,valLabelPath):
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

    for lineImage in tqdm(originImageFile.readlines()):
        imageIndex += 1
        if lineImage.strip().split("/")[0] in valID:
            valImageFile.write(lineImage)
            valImageIndex += 1
        else:
            trainImageFile.write(lineImage)
            trainImageIndex += 1

    for lineLabel in tqdm(originLabelFile.readlines()):
        labelIndex += 1
        if lineLabel.strip().split(" ")[0] in valID:
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

def generateTrainTest(imgDir, trainImagePath, valImagePath, trainSaveDir, valSaveDir):
    """
    :param imgDir:
    :param trainImagePath:
    :param valImagePath:
    :param trainSaveDir:
    :param valSaveDir:
    :return:
    """
    assert os.path.exists(imgDir),"{} is null !!!".format(imgDir)
    assert os.path.exists(trainImagePath),"{} is null !!!".format(trainImagePath)
    assert os.path.exists(valImagePath),"{} is null !!!".format(valImagePath)

    if not os.path.exists(trainSaveDir):
        os.makedirs(trainSaveDir)
    if not os.path.exists(valSaveDir):
        os.makedirs(valSaveDir)
    trainImageFile = open(trainImagePath,"r")
    valImageFile = open(valImagePath,"r")

    indexT = 0
    indexTImage = 0
    indexV = 0
    indexVImage = 0
    for line in tqdm(trainImageFile.readlines()):
        info = line.strip().split("/")[0]
        idPath = os.path.join(imgDir,info)
        # print("idPath=",idPath)
        tmpDir = os.path.join(trainSaveDir,info)
        if not os.path.exists(tmpDir):
            os.makedirs(tmpDir)
        for imageName in os.listdir(idPath):
            imagePath = os.path.join(idPath,imageName)
            # print("imagePath=",imagePath)
            shutil.copyfile(imagePath,tmpDir+"/"+imageName)
            indexTImage += 1
        indexT += 1


    for line in tqdm(valImageFile.readlines()):
        info = line.strip().split("/")[0]
        idPath = os.path.join(imgDir,info)
        # print("idPath=", idPath)
        tmpDir = os.path.join(valSaveDir, info)
        if not os.path.exists(tmpDir):
            os.makedirs(tmpDir)
        for imageName in os.listdir(idPath):
            imagePath = os.path.join(idPath, imageName)
            shutil.copyfile(imagePath, tmpDir+"/"+imageName)
            indexVImage += 1
        indexV += 1

    print("indexT=",indexT)
    print("indexTImage=",indexTImage)
    print("indexV=",indexV)
    print("indexVImage=",indexVImage)

def generateTrainTestTwo(imgDir, trainSaveDir, valSaveDir):
    """
    :param imgDir:
    :param trainSaveDir:
    :param valSaveDir:
    :return:
    """
    assert os.path.exists(imgDir),"{} is null !!!".format(imgDir)

    if not os.path.exists(trainSaveDir):
        os.makedirs(trainSaveDir)
    if not os.path.exists(valSaveDir):
        os.makedirs(valSaveDir)

    index = 0
    indexT = 0
    indexV = 0

    for idName in tqdm(os.listdir(imgDir)):
        index += 1
        idDirPath = os.path.join(imgDir,idName)
        if index <= 90000:
            shutil.copytree(idDirPath,trainSaveDir + "/" + idName)
            indexT += 1
        else:
            shutil.copytree(idDirPath, valSaveDir + "/" + idName)
            indexV += 1
    print("index=",index)
    print("indexT=",indexT)
    print("indexV=",indexV)

# if __name__  == "__main__":
#     imgDir = "/home/zhex/data/imgs_glintasia"
#     originImagePath = "annotationNewID/originImage.txt"
#     originLabelPath = "annotationNewID/originLabel.txt"
#     trainImagePath = "annotationNewID/trainImage.txt"
#     trainLabelPath = "annotationNewID/trainLabel.txt"
#     valImagePath = "annotationNewID/valImage.txt"
#     valLabelPath = "annotationNewID/valLabel.txt"
#     # splitDateset(imgDir,originImagePath,originLabelPath,trainImagePath,trainLabelPath,valImagePath,valLabelPath)
#     splitDatesetNewID(imgDir,originImagePath,originLabelPath,trainImagePath,trainLabelPath,valImagePath,valLabelPath)

if __name__ == "__main__":
    imgDir = "/home/zhex/data/imgs_glintasia"
    trainSaveDir = "/home/zhex/data/profileAsia/train"
    valSaveDir = "/home/zhex/data/profileAsia/val"
    generateTrainTestTwo(imgDir, trainSaveDir, valSaveDir)