#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : extract_feature_profile.py
# @time    : 8/6/20 9:57 AM
# @desc    : 
'''
import torch
import cv2
import numpy as np
import os
from backbone.model_irse import IR_SE_101
import torch.nn.functional as F


def np2tensor(imgNp):
    """
    @param imgNp:
    @return:
    """
    imgTensor = torch.from_numpy(imgNp)
    inputTensor = imgTensor.unsqueeze(0).permute(0,3,1,2)
    return inputTensor

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
    @param yaw:
    @param device:
    @param tta:
    @return:
    """
    # pre-requisites
    assert (os.path.exists(img_root)), "{} is null...".format(img_root)
    assert (os.path.exists(model_root)), "{} is null...".format(model_root)
    print('Backbone Model Root:', model_root)

    # load image
    img = cv2.imread(img_root)

    # resize image to [112,112]
    imgBgr = cv2.resize(img, (112, 112),interpolation=cv2.INTER_CUBIC)
    imgRgb = cv2.cvtColor(imgBgr,cv2.COLOR_BGR2RGB)
    # load numpy to tensor
    flipped = torch.from_numpy(imgRgb)
    flipped = flipped.unsqueeze(0).permute(0,3,1,2).float()
    # print("flipped.size()=",flipped.size())
    # load backbone from a checkpoint
    print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)
    backbone.eval()  # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = backbone(flipped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(flipped.to(device)).cpu())

    #     np.save("features.npy", features)
    #     features = np.load("features.npy")

    return features

# if __name__ == "__main__":
#     imageDir = "testImage"
#     modelRoot = "/home/zhex/pre_models/AILuoGang/2020-07-06-05-55_IR_SE_101_Epoch_55_LOSS_0.001.pth"
#     net = IR_SE_101([112])
#     imageNames = os.listdir(imageDir)
#     imageNames.sort(key=lambda x:int(x[:-4]))
#     for name1 in imageNames:
#         imagePath1 = os.path.join(imageDir,name1)
#         for name2 in imageNames:
#             imagePath2 = os.path.join(imageDir,name2)
#             if imagePath1 != imagePath2:
#                 feature1 = extractFeature(imagePath1, net, modelRoot)
#                 feature2 = extractFeature(imagePath2, net, modelRoot)
#                 # print("feature1.size()=",feature1.size())
#                 # print("feature2.size()=",feature2.size())
#                 feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
#                 feature2 = F.normalize(feature2)
#                 distance = feature1.mm(feature2.t())
#                 distance = round(distance.item(), 4)
#                 print("{} and {}'s distance=".format(name1,name2), distance)

if __name__ == "__main__":
    imageDir = "testImage"
    modelRoot = "/home/zhex/pre_models/AILuoGang/2020-07-06-05-55_IR_SE_101_Epoch_55_LOSS_0.001.pth"
    net = IR_SE_101([112])
    imageNames = os.listdir(imageDir)
    imageNames.sort(key=lambda x:int(x[:-4]))
    length = len(imageNames)
    for i in range(length):
        imagePath1 = os.path.join(imageDir,imageNames[i])
        feature1 = extractFeature(imagePath1, net, modelRoot)
        for j in range(i+1,length):
            imagePath2 = os.path.join(imageDir,imageNames[j])
            feature2 = extractFeature(imagePath2, net, modelRoot)
            # print("feature1.size()=",feature1.size())
            # print("feature2.size()=",feature2.size())
            feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
            feature2 = F.normalize(feature2)
            distance = feature1.mm(feature2.t())
            distance = round(distance.item(), 4)
            print("{} and {}'s distance=".format(imageNames[i],imageNames[j]), distance)