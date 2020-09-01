#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : face_detection.py
# @time    : 7/16/20 11:17 AM
# @desc    : 
'''
import os
import cv2
import numpy as np
from tqdm import tqdm
from centerface.prjPython.centerface import CenterFace
from fixBox.minitools import fixBox,fix_box,crop

def faceDetectionCenterFace(img:np.array,landmarks=True):
    h, w = img.shape[:2]
    centerFace = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerFace(img, h, w, threshold=0.6)
    else:
        dets = centerFace(img, threshold=0.6)

    if len(dets) > 0:
        bbox = []
        scores = []
        for det in dets:
            boxes, score = det[:4], det[4]
            boxes = list(map(lambda x:int(x),boxes))
            score = round(score,3)
            boxes = fixBox(boxes)
            x1 = boxes[0]
            y1 = boxes[1]
            x2 = boxes[2]
            y2 = boxes[3]
            boxes = [x1,y1,x2,y2]
            bbox.append(boxes)
            scores.append(score)
        if len(bbox) > 0:
            # newImg = crop(img,bbox[0])
            newImg = fix_box(img,bbox[0])
            return newImg, bbox[0]
        else:
            return np.array([]),[]
    else:
        return np.array([]),[]


def faceDetectionCenterMutilFace(img: np.array, landmarks=True):
    h, w = img.shape[:2]
    centerFace = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerFace(img, h, w, threshold=0.3)
    else:
        dets = centerFace(img, threshold=0.3)

    imgs = {}
    if len(dets) > 0:
        bbox = []
        scores = []
        for det in dets:
            boxes, score = det[:4], det[4]
            boxes = list(map(lambda x: int(x), boxes))
            score = round(score, 3)
            boxes = fixBox(boxes)
            x1 = boxes[0]
            y1 = boxes[1]
            x2 = boxes[2]
            y2 = boxes[3]
            boxes = [x1, y1, x2, y2]
            bbox.append(boxes)
            scores.append(score)
        if len(bbox) > 0:
            for singleBox in bbox:
                newImg = crop(img, singleBox)
                imgs[tuple(singleBox)] = newImg
            return imgs
        else:
            return imgs
    else:
        return imgs

if __name__ == "__main__":
    image_dir = "/home/zhex/data/818capture/pic2"
    save_dir = "/home/zhex/data/monitor/images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    index = 18374 ## pic2
    for name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir,name)
        try:
            image = cv2.imread(image_path)
            img_dict = faceDetectionCenterMutilFace(image)
            for key,value in img_dict.items():
                index += 1
                # cv2.imshow("img",value)
                # cv2.waitKey(10000)
                cv2.imwrite(save_dir + "/" + "{}.png".format(index),value)
        except Exception as e:
            print(e)
    print("index=",index)


# if __name__ == "__main__":
#     imageDir = "/home/zhex/Documents/project/profileFace/pic/AI"
#     for idName in os.listdir(imageDir):
#         idDir = os.path.join(imageDir,idName)
#         imageNames = os.listdir(idDir)
#         imageNames.sort(key=lambda x:int(x[:-4]))
#         saveDir = os.path.join("/home/zhex/Documents/project/profileFace/pic","new",idName)
#         if not os.path.exists(saveDir):
#             os.makedirs(saveDir)
#         index = 0
#         for imageName in imageNames:
#             imagePath = os.path.join(imageDir,idName,imageName)
#             print("imagePath=",imagePath)
#             image = cv2.imread(imagePath)
#             newImg,_ = faceDetectionCenterFace(image)
#             # print("newImg.shape=",newImg.shape)
#             cv2.imwrite(saveDir + "/" + "{}.png".format(index),newImg)
#             index += 1

