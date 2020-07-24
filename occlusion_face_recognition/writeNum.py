#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : writeNum.py
# @time    : 7/24/20 10:47 AM
# @desc    : 
'''
import os
def num(trainLabelPath, valLabelPath):
    """
    @param trainLabelPath:
    @param valLabelPath:
    @return:
    """
    assert os.path.exists(trainLabelPath),"{} is null !!!".format(trainLabelPath)
    assert os.path.exists(valLabelPath),"{} is null !!!".format(valLabelPath)
    trainLabelFile = open(trainLabelPath,"r")
    valLabelFile = open(valLabelPath,"r")
    indexT = 0
    indexV = 0
    TID = set()
    VID = set()
    for line in trainLabelFile.readlines():
        trainID = line.strip().split(" ")[0]
        TID.add(trainID)
        indexT += 1
    print("indexT=",indexT)
    print("len(TID)=",len(TID))

    for line in valLabelFile.readlines():
        valID = line.strip().split(" ")[0]
        VID.add(valID)
        indexV += 1
    print("indexV=", indexV)
    print("len(VID)=", len(VID))


if __name__ == "__main__":
    trainLabelPath = "/home/zhex/data/profiledataset/trainLabel.txt"
    valLabelPath = "/home/zhex/data/profiledataset/valLabel.txt"
    num(trainLabelPath,valLabelPath)
