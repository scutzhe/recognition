#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : test_val_data.py
# @time    : 7/25/20 11:14 AM
# @desc    : 
'''
import os
import cv2
import bcolz
import numpy as np

def get_val_pair(path, name):
    """
    @param path: val's dataset store path
    @param name: val's name
    @return:
    """
    carry = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r') # store data
    isSame = np.load('{}/{}_list.npy'.format(path, name))

    return carry, isSame

def get_val_pair_yaw(path, calculationWay, name):
    """
    @param path:
    @param calculationWay:
    @param name:
    @return:
    """
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r') # store data
    isSame = np.load('{}/{}_list.npy'.format(path, name))
    yaw = np.load('{}/yaw_{}_npy/{}_yaw.npy'.format(path, calculationWay, name))
    return carray, isSame, yaw


if __name__ == "__main__":
    valRootPath = "/home/zhex/data/evoLVe_val_dataset"
    valDatasetName = "lfw"
    carry, isSame, yaw = get_val_pair_yaw(valRootPath,"image",valDatasetName)
    print("carry.shape=",carry.shape) #(12000,3,112,112)
    print("isSame.shape=",isSame.shape) #(6000)
    # print("carry[0]=",carry[0].shape)
    print("yaw.shape=",yaw.shape)
    # for (i,yaw) in zip(range(carry.shape[0]),yaw):
    #     print("yaw=",yaw)
    #     image = carry[i]
    #     image = image.transpose(1,2,0)
    #     image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    #     cv2.imshow("image",image)
    #     cv2.waitKey(500)

# if __name__ == "__main__":
#     valRootPath = "/home/zhex/data/evoLVe_val_dataset"
#     valDatasetName = "lfw"
#     carry, isSame = get_val_pair(valRootPath,valDatasetName)
#     # print("carry.shape=",carry.shape) #(12000,3,112,112)
#     # print("isSame.shape=",isSame.shape) #(6000)
#     # print("carry[0]=",carry[0].shape)
#     for i in range(carry.shape[0]):
#         image = carry[i]
#         image = image.transpose(1,2,0)
#         # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
#         cv2.imshow("image",image)
#         cv2.waitKey(500)

# if __name__ == "__main__":
#     imagePath = "testImage/0.jpg"
#     img = cv2.imread(imagePath)
#     print("img=",img)
#     cv2.imshow("img",img)
#     cv2.waitKey(1000)