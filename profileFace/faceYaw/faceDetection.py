#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : faceDetection.py
# @time    : 7/16/20 11:17 AM
# @desc    : 
'''
import cv2
import numpy as np
from centerface.prjPython.centerface import CenterFace
from fixBox.minitools import fixBox,crop

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
            newImg = crop(img,bbox[0])
            return newImg, bbox[0]
        else:
            return np.array([]),[]
    else:
        return np.array([]),[]


def faceDetectionCenterMutilFace(img: np.array, landmarks=True):
    h, w = img.shape[:2]
    centerFace = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerFace(img, h, w, threshold=0.6)
    else:
        dets = centerFace(img, threshold=0.6)

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
    # imgPath = "testImage/1.jpg"
    # img = cv2.imread(imgPath)
    # newImg = faceDetectionCenterFace(img)
    # cv2.imshow("newImg",newImg)
    # cv2.waitKey(1000)

    videoPath = "/home/zhex/Videos/profileFace/20200713174000.avi"
    vid = cv2.VideoCapture(videoPath)
    while True:
        flag, frame = vid.read()
        if flag:
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            newImg = faceDetectionCenterFace(img)
            cv2.imshow("img",newImg)
            cv2.waitKey(1)