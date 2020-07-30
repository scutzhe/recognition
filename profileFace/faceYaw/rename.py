#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : rename.py
# @time    : 7/23/20 4:50 PM
# @desc    : 
'''
import os
import shutil
from tqdm import tqdm
def renames(imageDir,saveDir):
    """
    :param imageDir:
    :param saveDir:
    :return:
    """
    assert os.path.exists(imageDir), "{} is null!!!".format(imageDir)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    ids = os.listdir(imageDir)
    index = 0
    for idName in tqdm(ids):
        idOldPath = os.path.join(imageDir,idName)
        newName = str(index)
        idNewPath = os.path.join(saveDir,newName)
        shutil.copytree(idOldPath,idNewPath)
        index += 1
    print("index=",index)

def renamesTwo(imageDir):
    """
    :param imageDir:
    :return:
    """
    assert os.path.exists(imageDir), "{} is null!!!".format(imageDir)
    ids = os.listdir(imageDir)
    index = 0
    for idName in tqdm(ids):
        idOldPath = os.path.join(imageDir,idName)
        newName = str(index)
        idNewPath = os.path.join(imageDir,newName)
        os.rename(idOldPath,idNewPath)
        index += 1

        indexSub = 0
        for imageName in os.listdir(idNewPath):
            imagePath = os.path.join(idNewPath,imageName)
            newImageName = str(indexSub)
            imagePathNew = os.path.join(idNewPath,newImageName)
            os.rename(imagePath,imagePathNew)
            indexSub += 1
    print("index=",index)


if __name__ == "__main__":
    trainImageDir = "/home/zhex/data/profileAsia/train"
    trainSaveDir = "/home/zhex/data/profileNew/train"
    valImageDir = "/home/zhex/data/profileAsia/val"
    valSaveDir = "/home/zhex/data/profileNew/val"
    # renames(trainImageDir,trainSaveDir)
    renames(valImageDir,valSaveDir)