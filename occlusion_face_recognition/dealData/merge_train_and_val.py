#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : merge_train_and_val.py
# @time    : 8/5/20 8:27 AM
# @desc    : 
'''
import os
import shutil
from tqdm import tqdm
def renameZhe(imageDir, targetDir):
    """
    @param imageDir:
    @param targetDir:
    @return:
    """
    assert os.path.exists(imageDir),"{} is null!!!".format(imageDir)
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    IDs = os.listdir(imageDir)
    IDs.sort(key=lambda x:int(x))
    startIndex = 75000
    for idName in tqdm(IDs):
        idOldPath = os.path.join(imageDir,idName)
        idNewPath = os.path.join(targetDir,str(startIndex))
        os.rename(idOldPath,idNewPath)
        startIndex +=1
    print("startIndex=",startIndex)


def removeZhe(imageDir):
    """
    @param imageDir:
    @return:
    """
    assert os.path.exists(imageDir), "{} is null !!!".format(imageDir)
    IDs = os.listdir(imageDir)
    IDs.sort(key=lambda x: int(x))
    tmp = IDs[75000:]
    for idName in tqdm(tmp):
        idPath = os.path.join(imageDir,idName)
        shutil.rmtree(idPath)


def copyZhe(sourceDir,targetDir):
    """
    @param sourceDir:
    @param targetDir:
    @return:
    """
    assert os.path.exists(sourceDir),"{} is null !!!".format(sourceDir)
    IDs = os.listdir(sourceDir)
    IDs.sort(key=lambda x:int(x))
    for idName in tqdm(IDs):
        srcPath = os.path.join(sourceDir,idName)
        dstPath = os.path.join(targetDir,idName)
        shutil.copytree(srcPath,dstPath)


def changeLabel(labelPath,newLabelPath):
    """
    @param labelPath:
    @param newLabelPath:
    @return:
    """
    assert os.path.exists(labelPath),"{} is null !!!".format(labelPath)
    newLabelFile = open(newLabelPath,"a")
    label = open(labelPath,"r")
    next(label)
    line = label.readline()
    index = 0
    while line:
        info = line.strip().split(" ")
        ID = int(info[0]) + 75000
        yaw = info[1]
        newLabelFile.write(str(ID) + " " + yaw + "\n")
        line = label.readline()
        index += 1
    print("index=",index)

def changeImage(imagePath,newImagePath):
    """
    @param imagePath:
    @param newImagePath:
    @return:
    """
    assert os.path.exists(imagePath),"{} is null !!!".format(imagePath)
    newImageFile = open(newImagePath,"a")
    image = open(imagePath,"r")
    line = image.readline()
    index = 0
    while line:
        info = line.strip().split("/")
        ID = int(info[0]) + 75000
        imageName = info[1]
        newImageFile.write(str(ID) + "/" + imageName + "\n")
        line = image.readline()
        index += 1
    print("index=",index)

def checkLabel(filePath):
    """
    @param filePath:
    @return:
    """
    assert os.path.exists(filePath),"{} is null !!!".format(filePath)
    labelFile = open(filePath,"r")
    next(labelFile)
    line = labelFile.readline()
    id = set()
    index = 0
    while line:
        info = line.strip().split(" ")
        ID = info[0]
        id.add(ID)
        line = labelFile.readline()
        index += 1
    print("len(id)Label=",len(id))
    print("indexLabel=",index)

def checkImage(filePath):
    """
    @param filePath:
    @return:
    """
    assert os.path.exists(filePath),"{} is null !!!".format(filePath)
    imageFile = open(filePath,"r")
    line = imageFile.readline()
    id = set()
    index = 0
    while line:
        info = line.strip().split("/")
        ID = info[0]
        id.add(ID)
        line = imageFile.readline()
        index += 1
    print("len(id)Image=",len(id))
    print("indexImage=",index)

if __name__ == "__main__":
    ## step 1
    # imageDir = "/home/zhex/data/profiledataset/profileAsia/val"
    # targetDir = "/home/zhex/data/profiledataset/profileAsia/val_new"
    # renameZhe(imageDir,targetDir)

    ## tmpstep
    # imageDir = "/home/zhex/data/profiledatasetNoval/profileAsia/train"
    # removeZhe(imageDir)

    ## step 2
    # sourceDir = "/home/zhex/data/profiledataset/profileAsia/val_new"
    # targetDir = "/home/zhex/data/profiledataset/profileAsia/train"
    # copyZhe(sourceDir,targetDir)

    ## step3
    # labelPath = "/home/zhex/data/profiledataset/valLabel.txt"
    # newLabelPath = "/home/zhex/data/profiledataset/newValLabel.txt"
    # changeLabel(labelPath,newLabelPath)

    ## step4
    # imagePath = "/home/zhex/data/profiledataset/valImage.txt"
    # newImagePath = "/home/zhex/data/profiledataset/newValImage.txt"
    # changeImage(imagePath,newImagePath)

    ## step5
    imagePath ="/home/zhex/data/profiledataset/image.txt"
    labelPath ="/home/zhex/data/profiledataset/label.txt"
    checkLabel(labelPath)
    checkImage(imagePath)
