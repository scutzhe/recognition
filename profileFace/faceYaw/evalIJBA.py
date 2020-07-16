#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : evalIJBA.py
# @time    : 7/10/20 9:53 AM
# @desc    : calculation precision
'''
import os

def getGtYaw(txtPath,gtTxtPath):
    """
    @param txtPath:
    @param gtTxtPath:
    @return:
    """
    assert os.path.exists(txtPath),"{} is not exists !!!".format(txtPath)
    assert os.path.exists(gtTxtPath),"{} is not exists !!!".format(gtTxtPath)
    f = open(txtPath,"r")
    gtTxt = open(gtTxtPath,"a")
    for line in f.readlines():
        info = line.strip().split(" ")
        try:
            imgPath = info[0].split("/")[-2] + "/" + info[0].split("/")[-1]
            yaw = info[1]
            gtTxt.write(imgPath + " " + str(yaw) + "\n")
        except Exception as e:
            print(e)

def common(preTxtPath,gtTxtPath):
    """
    @param preTxtPath:
    @param gtTxtPath:
    @return:
    """
    assert os.path.exists(preTxtPath),"{} is not exists !!!".format(preTxtPath)
    assert os.path.exists(gtTxtPath),"{} is not exists !!!".format(gtTxtPath)
    commonNames = []
    tmpNames = []
    preTxt = open(preTxtPath,"r")
    gtTxt = open(gtTxtPath,"r")
    for line in preTxt.readlines():
        name = line.split(" ")[0]
        tmpNames.append(name)
    for line in gtTxt.readlines():
        name = line.split(" ")[0]
        if name in tmpNames:
            commonNames.append(name)
    preTxt.close()
    gtTxt.close()
    return commonNames

def compare(preTxtPath,gtTxtPath):
    """
    @param preTxtPath:
    @param gtTxtPath:
    @return:
    """
    commonNames = common(preTxtPath,gtTxtPath)
    length = len(commonNames)
    print("length=",length) # 4225
    yawPre = {}
    yawGt = {}
    num = 0
    preTxt = open(preTxtPath, "r")
    gtTxt = open(gtTxtPath, "r")
    for linePre in preTxt.readlines():
        namePre = linePre.split(" ")[0]
        if namePre in commonNames:
            yawPre[namePre] = (float(linePre.split(" ")[1]))
    for lineGt in gtTxt.readlines():
        nameGt = lineGt.split(" ")[0]
        if nameGt in commonNames:
            yawGt[nameGt] = float(lineGt.split(" ")[1])
    preTxt.close()
    gtTxt.close()

    for keyPre in yawPre.keys():
        for keyGt in yawGt.keys():
            if keyPre == keyGt:
                diff = yawPre[keyPre] - yawGt[keyGt]
                error = diff / yawGt[keyGt]
                if abs(error) < 0.05:
                    num += 1
                    print("num=", num)
    precision = round(num / length,5)
    return precision

if __name__ == "__main__":
    rootPath = "/home/zhex/data/IJBA/align_image_11/ijb_a_11_align_split1"
    ### gtTxt
    # gtTxt = open("gtTxt.txt","a")
    # for txt in os.listdir(rootPath):
    #     if txt.endswith(".txt"):
    #         if txt.split(".")[0] in ["frame_list_linear","img_list_linear"]:
    #             txtPath = os.path.join(rootPath,txt)
    #             print("txtPath=",txtPath)
    #             getGtYaw(txtPath,gtTxt)
    ### precision
    gtTxtPath= "yawResult/gtTxt.txt"
    preTxtPath = "yawResult/preTxt.txt"
    precision = compare(preTxtPath,gtTxtPath)
    print("precision=",precision)




