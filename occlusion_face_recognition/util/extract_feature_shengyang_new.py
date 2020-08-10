#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : extract_feature_shengyang_new.py
# @time    : 8/7/20 5:15 PM
# @desc    : 
'''

# Helper function for extracting features from pre-trained models
import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from backbone.model_irse import IR_SE_101
import torch.nn.functional as F


def l2_norm(input, axis=1):
    """
    @param input:
    @param axis:
    @return:
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

def extractFeature(img_root, backbone, model_root,
                   device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta=True):
    """
    @param img_root:
    @param backbone:
    @param model_root:
    @param device:
    @param tta:
    @return:
    """
    # pre-requisites
    assert (os.path.exists(img_root))
    # print('Testing Data Root:', img_root)
    assert (os.path.exists(model_root))
    # print('Backbone Model Root:', model_root)

    # load image
    img = cv2.imread(img_root)
    ccropped = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
    ccropped = ccropped[..., ::-1]  # BGR to RGB

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype=np.float32)
    ccropped = (ccropped - 127.5) / 128
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype=np.float32)
    flipped = (flipped - 127.5) / 128
    flipped = torch.from_numpy(flipped)

    # load backbone from a checkpoint
    # print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)

    # extract features
    backbone.eval()  # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())

    #     np.save("features.npy", features)
    #     features = np.load("features.npy")

    return features

# if __name__ == "__main__":
#     imageDir = "testImage"
#     modelRoot = "/home/zhex/pre_models/AILuoGang/2020-07-06-05-55_IR_SE_101_Epoch_55_LOSS_0.001.pth"
#     net = IR_SE_101([112])
#     imageNames = os.listdir(imageDir)
#     imageNames.sort(key=lambda x: int(x[:-4]))
#     length = len(imageNames)
#     for i in range(length):
#         imagePath1 = os.path.join(imageDir,imageNames[i])
#         feature1 = extractFeature(imagePath1, net, modelRoot)
#         # feature1 = extract_feature(imagePath1, net, modelRoot, yaw[i])
#         for j in range(i+1,length):
#             imagePath2 = os.path.join(imageDir,imageNames[j])
#             # feature2 = extract_feature(imagePath2, net, modelRoot,yaw[j])
#             feature2 = extractFeature(imagePath2, net, modelRoot)
#             # print("feature1.size()=",feature1.size())
#             # print("feature2.size()=",feature2.size())
#             distance = feature1.mm(feature2.t())
#             distance = round(distance.item(), 4)
#             print("{} and {}'s distance=".format(imageNames[i],imageNames[j]), distance)


# if __name__ == "__main__":
#     imageDirGZ = "/home/zhex/Documents/project/profileFace/pic/AI/zhengxiangzhong"
#     imageDirFZ = "/home/zhex/Videos/profileFace/monitor/xiangzhong/frontal"
#     imageDirPZ = "/home/zhex/Videos/profileFace/monitor/xiangzhong/profile"
#     modelRoot = "/home/zhex/pre_models/AILuoGang/2020-07-06-05-55_IR_SE_101_Epoch_55_LOSS_0.001.pth"
#     net = IR_SE_101([112])
#     imageNamesGZ = os.listdir(imageDirGZ)
#     imageNamesGZ.sort(key=lambda x:int(x[:-4]))
#     lengthGZ = len(imageNamesGZ)
#
#     imageNamesFZ = os.listdir(imageDirFZ)
#     imageNamesFZ.sort(key=lambda x:int(x[:-4]))
#     lengthFZ = len(imageNamesFZ)
#
#     imageNamesPZ = os.listdir(imageDirPZ)
#     imageNamesPZ.sort(key=lambda x: int(x[:-4]))
#     lengthPZ = len(imageNamesPZ)
#
#
#     for i in range(lengthGZ):
#         imagePath1 = os.path.join(imageDirGZ,imageNamesGZ[i])
#         feature1 = extractFeature(imagePath1, net, modelRoot)
#         for j in range(lengthFZ):
#             imagePath2 = os.path.join(imageDirFZ,imageNamesFZ[j])
#             feature2 = extractFeature(imagePath2, net, modelRoot)
#             # print("feature1.size()=",feature1.size())
#             # print("feature2.size()=",feature2.size())
#             feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
#             feature2 = F.normalize(feature2)
#             distance = feature1.mm(feature2.t())
#             distance = round(distance.item(), 4)
#             print("{} and {}'s distance=".format(imageNamesGZ[i],imageNamesFZ[j]), distance)

# if __name__ == "__main__":
#     imageDirGZ = "/home/zhex/Documents/project/profileFace/pic/AI/lingangxin"
#     imageDirFZ = "/home/zhex/Videos/profileFace/monitor/gangxin/frontal"
#     imageDirPZ = "/home/zhex/Videos/profileFace/monitor/gangxin/profile"
#     modelRoot = "/home/zhex/pre_models/AILuoGang/2020-07-06-05-55_IR_SE_101_Epoch_55_LOSS_0.001.pth"
#     net = IR_SE_101([112])
#     imageNamesGZ = os.listdir(imageDirGZ)
#     imageNamesGZ.sort(key=lambda x:int(x[:-4]))
#     lengthGZ = len(imageNamesGZ)
#
#     imageNamesFZ = os.listdir(imageDirFZ)
#     imageNamesFZ.sort(key=lambda x:int(x[:-4]))
#     lengthFZ = len(imageNamesFZ)
#
#     imageNamesPZ = os.listdir(imageDirPZ)
#     imageNamesPZ.sort(key=lambda x: int(x[:-4]))
#     lengthPZ = len(imageNamesPZ)
#
#
#     for i in range(lengthGZ):
#         imagePath1 = os.path.join(imageDirGZ,imageNamesGZ[i])
#         feature1 = extractFeature(imagePath1, net, modelRoot)
#         for j in range(lengthFZ):
#             imagePath2 = os.path.join(imageDirFZ,imageNamesFZ[j])
#             feature2 = extractFeature(imagePath2, net, modelRoot)
#             # print("feature1.size()=",feature1.size())
#             # print("feature2.size()=",feature2.size())
#             feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
#             feature2 = F.normalize(feature2)
#             distance = feature1.mm(feature2.t())
#             distance = round(distance.item(), 4)
#             print("{} and {}'s distance=".format(imageNamesGZ[i],imageNamesFZ[j]), distance)

# if __name__ == "__main__":
#     # imageDirF = "/home/zhex/Videos/profileFace/monitor/xiangzhong/frontal"
#     imageDirF = "/home/zhex/Videos/profileFace/monitor/gangxin/frontal"
#     # imageDirP = "/home/zhex/Videos/profileFace/monitor/xiangzhong/profile"
#     imageDirP = "/home/zhex/Videos/profileFace/monitor/gangxin/profile"
#     modelRoot = "/home/zhex/pre_models/AILuoGang/2020-07-06-05-55_IR_SE_101_Epoch_55_LOSS_0.001.pth"
#     net = IR_SE_101([112])
#     imageNamesF = os.listdir(imageDirF)
#     imageNamesF.sort(key=lambda x:int(x[:-4]))
#     length = len(imageNamesF)
#
#     imageNamesP = os.listdir(imageDirP)
#     imageNamesP.sort(key=lambda x: int(x[:-4]))
#
#
#     for i in range(length):
#         imagePath1 = os.path.join(imageDirF,imageNamesF[i])
#         feature1 = extractFeature(imagePath1, net, modelRoot)
#         for j in range(length):
#             imagePath2 = os.path.join(imageDirP,imageNamesP[j])
#             feature2 = extractFeature(imagePath2, net, modelRoot)
#             # print("feature1.size()=",feature1.size())
#             # print("feature2.size()=",feature2.size())
#             feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
#             feature2 = F.normalize(feature2)
#             distance = feature1.mm(feature2.t())
#             distance = round(distance.item(), 4)
#             print("{} and {}'s distance=".format(imageNamesF[i],imageNamesP[j]), distance)

if __name__ == "__main__":
    imageDirGZ = "/home/zhex/Documents/project/profileFace/pic/AI/zhengxiangzhong"
    imageDirGL = "/home/zhex/Documents/project/profileFace/pic/AI/lingangxin"
    modelRoot = "/home/zhex/pre_models/AILuoGang/2020-07-06-05-55_IR_SE_101_Epoch_55_LOSS_0.001.pth"
    net = IR_SE_101([112])

    imageNamesGZ = os.listdir(imageDirGZ)
    imageNamesGZ.sort(key=lambda x:int(x[:-4]))
    lengthGZ = len(imageNamesGZ)

    imageNamesGL = os.listdir(imageDirGL)
    imageNamesGL.sort(key=lambda x:int(x[:-4]))
    lengthGL = len(imageNamesGL)

    for i in range(lengthGZ):
        imagePath1 = os.path.join(imageDirGZ,imageNamesGZ[i])
        feature1 = extractFeature(imagePath1, net, modelRoot)
        for j in range(lengthGL):
            imagePath2 = os.path.join(imageDirGL,imageNamesGL[j])
            feature2 = extractFeature(imagePath2, net, modelRoot)
            # print("feature1.size()=",feature1.size())
            # print("feature2.size()=",feature2.size())
            feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
            feature2 = F.normalize(feature2)
            distance = feature1.mm(feature2.t())
            distance = round(distance.item(), 4)
            print("{} and {}'s distance=".format(imageNamesGZ[i],imageNamesGL[j]), distance)