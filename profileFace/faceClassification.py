#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : faceClassification.py
# @time    : 8/4/20 10:13 AM
# @desc    : 
'''
import os
import cv2
from faceYaw.faceDetection import faceDetectionCenterFace
from face_classification.classification import FaceClass
faceClasser = FaceClass(model_path='/home/zhex/pre_models/AILuoGang/faceclassification.trt')

# if __name__ == "__main__":
#     imageDir = "faceYaw/testImage/"
#     imgs = []
#     for imageName in os.listdir(imageDir):
#         imagePath = os.path.join(imageDir,imageName)
#         image = cv2.imread(imagePath)
#         image = cv2.cvtColor(image,cv2.cv2.COLOR_BGR2RGB)
#         imgs.append(image)
#     labels = faceClasser.classify(imgs)
#     print("labels=",labels)

if __name__ == "__main__":
    videoPath = "/home/zhex/Videos/profileFace/hall/test/01.mp4"
    videoName = videoPath.split("/")[-1]
    vid = cv2.VideoCapture(videoPath)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(videoName,fourcc,10,(2560,1440))
    out = cv2.VideoWriter(videoName,fourcc,10,(1920,1080))
    while True:
        flag, frame = vid.read()
        image = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
        imageNew, box = faceDetectionCenterFace(image)
        if len(box) == 0:
            continue
        else:
            label = faceClasser.classifyNormal([imageNew])
            cv2.rectangle(frame, (2 * box[0], 2 * box[1]), (2 * box[2], 2 * box[3]), (0, 0, 255), 2)
            cv2.putText(frame, str(label), (2 * box[0], 2 * box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow("frame",frame)
            cv2.waitKey(1)
            out.write(frame)
