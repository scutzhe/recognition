#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : add_val_yaw.py
# @time    : 7/25/20 11:56 AM
# @desc    : 
'''

import os
import cv2
import bcolz
import numpy as np
from tqdm import tqdm
from PIL import Image
from demo import createModel, yawCoefficient

def angleNoDetection(img,model):
    """
    :param img:
    :param model:
    :return:
    """
    # angle(detected,img,faces,ad,img_size,img_w,img_h,model)
    imgNew = cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
    imgNew = np.expand_dims(imgNew,axis=0)
    p_result = model.predict(imgNew)
    yaw = p_result[0][0]
    # pitch = p_result[0][1]
    # roll = p_result[0][2]
    return yaw

def get_val_pair(path, name):
    """
    @param path: val's dataset store path
    @param name: val's name
    @return:
    """
    carry = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r') # store data
    isSame = np.load('{}/{}_list.npy'.format(path, name))

    return carry, isSame

def check(npyPath):
    """
    :param npyPath:
    :return:
    """
    assert os.path.exists(npyPath),"{} is null !!!".format(npyPath)
    npFile = np.load(npyPath)
    print("npFile=",npFile)
    print("npFile.shape=",npFile.shape)



## save origin pic
# if __name__ == "__main__":
#     ### create model
#     valRootPath = "/home/zhex/data/evoLVe_val_dataset"
#     for valDatasetName in os.listdir(valRootPath):
#         tmpDir = os.path.join(valRootPath,valDatasetName)
#         if os.path.isdir(tmpDir):
#             # print("tmpDir=",tmpDir)
#             # valDatasetName = "lfw"
#             carry, isSame = get_val_pair(valRootPath,valDatasetName)
#             # print("carry.shape=",carry.shape) #(12000,3,112,112)
#             # print("isSame.shape=",isSame.shape) #(6000)
#             # print("carry[0]=",carry[0].shape)
#             saveDir = "valImage" + "/" +valDatasetName
#             # print("saveDir=",saveDir)
#             if not os.path.exists(saveDir):
#                 os.makedirs(saveDir)
#             for i in tqdm(range(carry.shape[0])):
#                 image = carry[i]
#                 image = image.transpose(1,2,0)
#                 image = 255.0 * image
#                 cv2.imwrite(saveDir + "/" + "{}.png".format(i),image)

#  coefficient yaw
# if __name__ == "__main__":
#     ### create model
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1, weightPath2, weightPath3)
#
#     valRootPath = "/home/zhex/data/evoLVe_val_dataset"
#     for valDatasetName in os.listdir(valRootPath):
#         tmpDir = os.path.join(valRootPath,valDatasetName)
#         if os.path.isdir(tmpDir):
#             # print("tmpDir=",tmpDir)
#             # valDatasetName = "lfw"
#             carry, isSame = get_val_pair(valRootPath,valDatasetName)
#             # print("carry.shape=",carry.shape) #(12000,3,112,112)
#             # print("isSame.shape=",isSame.shape) #(6000)
#             # print("carry[0]=",carry[0].shape)
#             sum = []
#             for i in tqdm(range(carry.shape[0])):
#                 image = carry[i]
#                 image = image.transpose(1,2,0)
#                 yaw = angleNoDetection(image, model)
#                 coefficient = yawCoefficient(abs(yaw))
#                 # print("coefficient=",coefficient)
#                 sum.append(coefficient)
#             npSave = np.array(sum)
#             np.save("yaw_npy" + "/" + "{}_yaw.npy".format(valDatasetName),npSave)

# if __name__ == "__main__":
#     ### create model
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1, weightPath2, weightPath3)
#
#     valRootPath = "/home/zhex/data/evoLVe_val_dataset"
#     valDatasetName = "lfw"
#     carry, isSame = get_val_pair(valRootPath, valDatasetName)
#     # print("carry.shape=",carry.shape) #(12000,3,112,112)
#     # print("isSame.shape=",isSame.shape) #(6000)
#     # print("carry[0]=",carry[0].shape)
#     sum = []
#     for i in tqdm(range(carry.shape[0])):
#         image = carry[i]
#         image = image.transpose(1, 2, 0)
#         # cv2.imshow("image",image)
#         # cv2.waitKey(1)
#         cv2.imwrite("valImage" + "/" + "{}.png".format(i),image*255)
#         yaw = angleNoDetection(image, model)
#         coefficient = yawCoefficient(abs(yaw))
#         print("coefficient=", coefficient)
#         sum.append(coefficient)
#     npSave = np.array(sum)
#     np.save(valDatasetName + ".npy", npSave)

if __name__ == "__main__":
    ### create model
    weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model = createModel(weightPath1, weightPath2, weightPath3)

    valRootPath = "/home/zhex/data/evoLVe_val_image"
    savePath = "yaw_image_npy"
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    for datasetName in os.listdir(valRootPath):
        datasetDir = os.path.join(valRootPath,datasetName)
        names = os.listdir(datasetDir)
        names.sort(key=lambda x:int(x[:-4]))
        # print("names=",names)
        sum = []
        for imageName in tqdm(names):
            imagePath = os.path.join(datasetDir,imageName)
            # print("imagePath=",imagePath)
            image = cv2.imread(imagePath)
            imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            yaw = angleNoDetection(imageRGB, model)
            coefficient = yawCoefficient(abs(yaw))
            # print("coefficient=",coefficient)
            sum.append(coefficient)
        npSave = np.array(sum)
        print("npSave.shape=",npSave.shape)
        np.save(savePath + "/" +"{}_yaw.npy".format(datasetName), npSave)
