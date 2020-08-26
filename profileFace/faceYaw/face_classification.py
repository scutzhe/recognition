#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : face_classification.py
# @time    : 8/4/20 10:13 AM
# @desc    : 
'''
import os
import cv2
import shutil
from tqdm import tqdm
from torchvision.transforms import transforms
from face_detection import faceDetectionCenterFace
from face_classification.classification import FaceClass
from config.settings import TRAIN_MEAN,TRAIN_STD
faceClasser = FaceClass(model_path='/home/zhex/pre_models/AILuoGang/faceclassification.trt')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
])


if __name__ == "__main__":
    image_dir = "/home/zhex/Downloads/one_occ_dir"
    save_frontal_dir = "/home/zhex/Downloads/occ/frontal"
    save_profile_dir = "/home/zhex/Downloads/occ/profile"
    if not os.path.exists(save_frontal_dir):
        os.makedirs(save_frontal_dir)
    if not os.path.exists(save_profile_dir):
        os.makedirs(save_profile_dir)
    indexF = 0
    indexP = 0
    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir,image_name)
        image_BGR = cv2.imread(image_path)
        # H, W = image_BGR.shape[:2]
        # image_face_BGR = image_BGR[H//4 : 3 * H//4,W//4 : 3 * W//4, :]
        image_RGB = cv2.cvtColor(image_BGR,cv2.COLOR_BGR2RGB)
        label = faceClasser.classifyNormal([image_RGB])
        if label == 0:
            shutil.copy(image_path,save_frontal_dir)
            indexF += 1
        elif label == 1:
            shutil.copy(image_path,save_profile_dir)
            indexP += 1
    print("indexF=",indexF)
    print("indexP=",indexP)

# if __name__ == "__main__":
#     videoPath = "/home/zhex/Videos/profileFace/luogang/4m/戴口罩_h4米d1米_d2米_d3米.mp4"
#     videoName = videoPath.split("/")[-1]
#     vid = cv2.VideoCapture(videoPath)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(videoName,fourcc,10,(2560,1440))
#     # out = cv2.VideoWriter(videoName,fourcc,10,(1920,1080))
#     while True:
#         flag, frame = vid.read()
#         if flag:
#             try:
#                 image = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
#                 imageBGR, box = faceDetectionCenterFace(image)
#                 imageRGB = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB) ###BGR->RGB
#                 if len(box) == 0:
#                     continue
#                 else:
#                     label = faceClasser.classifyNormal([imageRGB])
#                     print("label=",label)
#                     cv2.rectangle(frame, (2 * box[0], 2 * box[1]), (2 * box[2], 2 * box[3]), (0, 0, 255), 2)
#                     cv2.putText(frame, str(label), (2 * box[0], 2 * box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#                     # cv2.imshow("frame",frame)
#                     # cv2.waitKey(1)
#                     out.write(frame)
#             except Exception as e:
#                 print(e)
#         else:
#             break