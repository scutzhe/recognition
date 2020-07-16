#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : transformationCoefficient.py
# @time    : 7/13/20 7:15 PM
# @desc    : 
'''
import math
import numpy as np

def calCoefficient(radian):
    """
    :param radian:
    :return:
    """
    midValue = 4 * radian / np.pi - 1
    return 1/(1+np.exp(-midValue))

def coefficientToAngle(coefficient:float):
    """
    :param coefficient:
    :return:
    """
    return (np.pi / 4) * (1 + math.log(coefficient/(1-coefficient)))

def fun(coeff:float):
    """
    :param coeff:
    :return:
    """
    rad = coefficientToAngle(coeff)
    if rad > 0:
        return coeff
    else:
        return calCoefficient(-coeff)


def transform(trainLabelFile,traiLabelNewFile):
    """
    :param trainLabelFile:
    :return:
    """
    num = 0
    for line in trainLabelFile.readlines():
        num += 1
        info = line.split(" ")
        imgPath = info[0]
        coefficient = info[1]
        coefficientNum = float(coefficient)
        coefficientNew = fun(coefficientNum)
        traiLabelNewFile.write(imgPath + " " + str(coefficientNew) + "\n")
        print("num=",num)
    print("num=",num)



# if __name__ == "__main__":
#     testRadNeg = -np.pi / 4
#     tmpResNeg = calCoefficient(testRadNeg)
#     print("tmpResNeg=",tmpResNeg)
#
#     testRadPos = np.pi / 4
#     tmpResPos = calCoefficient(testRadPos)
#     print("tmpResPos=",tmpResPos)
#
#     radian = coefficientToAngle(tmpResNeg)
#     print("radian=",radian)
#
#     preRes = calCoefficient(-radian)
#     print("preRes=",preRes)

if __name__ == "__main__":
    trainLableOriginFile = open("tainLabel", "a")
    trainLabelNewFile = open("trainLabelNew.txt","a")
    transform(trainLableOriginFile,trainLabelNewFile)



