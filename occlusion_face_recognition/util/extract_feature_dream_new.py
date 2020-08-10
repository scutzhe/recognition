#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : extract_feature_dream_new.py
# @time    : 8/7/20 5:15 PM
# @desc    : 
'''

# Helper function for extracting features from pre-trained models
import torch
import cv2
import numpy as np
import os
from backbone.model_dream import IR_SE_DREAM_101
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

## refer to train coding
def extractFeature(img_root, backbone, model_root, yaw,
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

    # yaw
    yawNp = np.array([yaw])
    yaw_ = torch.tensor(yawNp).float()
    yawTensor = yaw_.unsqueeze(1)


    # load backbone from a checkpoint
    # print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)

    # extract features
    backbone.eval()  # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = backbone(flipped.to(device), yawTensor.to(device)).cpu() + backbone(flipped.to(device),yawTensor.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())

    #     np.save("features.npy", features)
    #     features = np.load("features.npy")

    return features

# if __name__ == "__main__":
#     imageDir = "testImage"
#     yawPath = "yaw_npy/yawTest.npy"
#     modelRoot = "model/2020-07-30-09-15_IR_SE_DREAM_101_Epoch_101_LOSS_0.005.pth"
#     net = IR_SE_DREAM_101([112])
#     imageNames = os.listdir(imageDir)
#     imageNames.sort(key=lambda x: int(x[:-4]))
#     length = len(imageNames)
#     yaw = np.load(yawPath)
#     for i in range(length):
#         imagePath1 = os.path.join(imageDir,imageNames[i])
#         feature1 = extractFeature(imagePath1, net, modelRoot, yaw[i])
#         # feature1 = extract_feature(imagePath1, net, modelRoot, yaw[i])
#         for j in range(i+1,length):
#             imagePath2 = os.path.join(imageDir,imageNames[j])
#             # feature2 = extract_feature(imagePath2, net, modelRoot,yaw[j])
#             feature2 = extractFeature(imagePath2, net, modelRoot,yaw[j])
#             # print("feature1.size()=",feature1.size())
#             # print("feature2.size()=",feature2.size())
#             distance = feature1.mm(feature2.t())
#             distance = round(distance.item(), 4)
#             print("{} and {}'s distance=".format(imageNames[i],imageNames[j]), distance)

# if __name__ == "__main__":
#     # imageDirF = "/home/zhex/Videos/profileFace/monitor/xiangzhong/frontal"
#     imageDirF = "/home/zhex/Videos/profileFace/monitor/gangxin/frontal"
#     # imageDirP = "/home/zhex/Videos/profileFace/monitor/xiangzhong/profile"
#     imageDirP = "/home/zhex/Videos/profileFace/monitor/gangxin/profile"
#     # yawF = "yaw_npy/yawMonitorFrontalzhengxiangzhong.npy"
#     yawF = "yaw_npy/yawMonitorFrontallingangxin.npy"
#     # yawP = "yaw_npy/yawMonitorProfilezhengxiangzhong.npy"
#     yawP = "yaw_npy/yawMonitorProfilelingangxin.npy"
#     modelRoot = "model/2020-07-30-09-15_IR_SE_DREAM_101_Epoch_101_LOSS_0.005.pth"
#     net = IR_SE_DREAM_101([112])
#     imageNamesF = os.listdir(imageDirF)
#     imageNamesF.sort(key=lambda x:int(x[:-4]))
#     yawNpF = np.load(yawF)
#     length = len(imageNamesF)
#
#     imageNamesP = os.listdir(imageDirP)
#     imageNamesP.sort(key=lambda x: int(x[:-4]))
#     yawNpP = np.load(yawP)
#
#
#     for i in range(length):
#         imagePath1 = os.path.join(imageDirF,imageNamesF[i])
#         feature1 = extractFeature(imagePath1, net, modelRoot, yawNpF[i])
#         for j in range(length):
#             imagePath2 = os.path.join(imageDirP,imageNamesP[j])
#             feature2 = extractFeature(imagePath2, net, modelRoot, yawNpP[j])
#             # print("feature1.size()=",feature1.size())
#             # print("feature2.size()=",feature2.size())
#             feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
#             feature2 = F.normalize(feature2)
#             distance = feature1.mm(feature2.t())
#             distance = round(distance.item(), 4)
#             print("{} and {}'s distance=".format(imageNamesF[i],imageNamesP[j]), distance)


# if __name__ == "__main__":
#     imageDirGZ = "/home/zhex/Documents/project/profileFace/pic/AI/zhengxiangzhong"
#     imageDirFZ = "/home/zhex/Videos/profileFace/monitor/xiangzhong/frontal"
#     imageDirPZ = "/home/zhex/Videos/profileFace/monitor/xiangzhong/profile"
#     yawGZ = "yaw_npy/yawGalleryZhengXiangZhong.npy"
#     yawFZ = "yaw_npy/yawFrontalZhengXiangZhong.npy"
#     yawPZ = "yaw_npy/yawProfileZhengXiangZhong.npy"
#     modelRoot = "model/2020-07-30-09-15_IR_SE_DREAM_101_Epoch_101_LOSS_0.005.pth"
#     net = IR_SE_DREAM_101([112])
#     imageNamesGZ = os.listdir(imageDirGZ)
#     imageNamesGZ.sort(key=lambda x:int(x[:-4]))
#     yawNpGZ = np.load(yawGZ)
#     lengthGZ = len(imageNamesGZ)
#
#     imageNamesFZ = os.listdir(imageDirFZ)
#     imageNamesFZ.sort(key=lambda x:int(x[:-4]))
#     yawNpFZ = np.load(yawFZ)
#     lengthFZ = len(imageNamesFZ)
#
#     imageNamesPZ = os.listdir(imageDirPZ)
#     imageNamesPZ.sort(key=lambda x: int(x[:-4]))
#     yawNpPZ = np.load(yawPZ)
#     lengthPZ = len(imageNamesPZ)
#
#
#     for i in range(lengthGZ):
#         imagePath1 = os.path.join(imageDirGZ,imageNamesGZ[i])
#         feature1 = extractFeature(imagePath1, net, modelRoot, yawNpGZ[i])
#         for j in range(lengthPZ):
#             imagePath2 = os.path.join(imageDirPZ,imageNamesPZ[j])
#             feature2 = extractFeature(imagePath2, net, modelRoot, yawNpPZ[j])
#             # print("feature1.size()=",feature1.size())
#             # print("feature2.size()=",feature2.size())
#             feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
#             feature2 = F.normalize(feature2)
#             distance = feature1.mm(feature2.t())
#             distance = round(distance.item(), 4)
#             print("{} and {}'s distance=".format(imageNamesGZ[i],imageNamesPZ[j]), distance)

# if __name__ == "__main__":
#     imageDirGZ = "/home/zhex/Documents/project/profileFace/pic/AI/lingangxin"
#     imageDirFZ = "/home/zhex/Videos/profileFace/monitor/gangxin/frontal"
#     imageDirPZ = "/home/zhex/Videos/profileFace/monitor/gangxin/profile"
#     yawGZ = "yaw_npy/yawGalleryLinGangXin.npy"
#     yawFZ = "yaw_npy/yawFrontalLinGangXin.npy"
#     yawPZ = "yaw_npy/yawProfileLinGangXin.npy"
#     modelRoot = "model/2020-07-30-09-15_IR_SE_DREAM_101_Epoch_101_LOSS_0.005.pth"
#     net = IR_SE_DREAM_101([112])
#     imageNamesGZ = os.listdir(imageDirGZ)
#     imageNamesGZ.sort(key=lambda x:int(x[:-4]))
#     yawNpGZ = np.load(yawGZ)
#     lengthGZ = len(imageNamesGZ)
#
#     imageNamesFZ = os.listdir(imageDirFZ)
#     imageNamesFZ.sort(key=lambda x:int(x[:-4]))
#     yawNpFZ = np.load(yawFZ)
#     lengthFZ = len(imageNamesFZ)
#
#     imageNamesPZ = os.listdir(imageDirPZ)
#     imageNamesPZ.sort(key=lambda x: int(x[:-4]))
#     yawNpPZ = np.load(yawPZ)
#     lengthPZ = len(imageNamesPZ)
#
#
#     for i in range(lengthGZ):
#         imagePath1 = os.path.join(imageDirGZ,imageNamesGZ[i])
#         feature1 = extractFeature(imagePath1, net, modelRoot, yawNpGZ[i])
#         for j in range(lengthPZ):
#             imagePath2 = os.path.join(imageDirPZ,imageNamesPZ[j])
#             feature2 = extractFeature(imagePath2, net, modelRoot, yawNpPZ[j])
#             # print("feature1.size()=",feature1.size())
#             # print("feature2.size()=",feature2.size())
#             feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
#             feature2 = F.normalize(feature2)
#             distance = feature1.mm(feature2.t())
#             distance = round(distance.item(), 4)
#             print("{} and {}'s distance=".format(imageNamesGZ[i],imageNamesPZ[j]), distance)

if __name__ == "__main__":
    imageDirGZ = "/home/zhex/Documents/project/profileFace/pic/AI/zhengxiangzhong"
    imageDirGL = "/home/zhex/Documents/project/profileFace/pic/AI/lingangxin"
    yawGZ= "yaw_npy/yawGalleryZhengXiangZhong.npy"
    yawGL= "yaw_npy/yawGalleryLinGangXin.npy"
    modelRoot = "model/2020-07-30-09-15_IR_SE_DREAM_101_Epoch_101_LOSS_0.005.pth"
    net = IR_SE_DREAM_101([112])

    imageNamesGZ = os.listdir(imageDirGZ)
    imageNamesGZ.sort(key=lambda x:int(x[:-4]))
    yawNpGZ = np.load(yawGZ)
    lengthGZ = len(imageNamesGZ)

    imageNamesGL = os.listdir(imageDirGL)
    imageNamesGL.sort(key=lambda x:int(x[:-4]))
    yawNpGL = np.load(yawGL)
    lengthGL = len(imageNamesGL)

    for i in range(lengthGZ):
        imagePath1 = os.path.join(imageDirGZ,imageNamesGZ[i])
        feature1 = extractFeature(imagePath1, net, modelRoot, yawNpGZ[i])
        for j in range(lengthGL):
            imagePath2 = os.path.join(imageDirGL,imageNamesGL[j])
            feature2 = extractFeature(imagePath2, net, modelRoot, yawNpGL[j])
            # print("feature1.size()=",feature1.size())
            # print("feature2.size()=",feature2.size())
            feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
            feature2 = F.normalize(feature2)
            distance = feature1.mm(feature2.t())
            distance = round(distance.item(), 4)
            print("{} and {}'s distance=".format(imageNamesGZ[i],imageNamesGL[j]), distance)
