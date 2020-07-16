#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : minitools.py
# @time    : 7/4/20 4:09 PM
# @desc    : 
'''
import cv2
import numpy as np
from config.settings import FACESIZE,FACEFACTOR
from fixBox.dealNum import integer

def fixBox(bbox:list)->list:
    boxes = []
    assert len(bbox) != 0, "bbox is empty"
    diff = (bbox[3] - bbox[1]) - (bbox[2] - bbox[0])
    if diff > 0:
        bbox[0] -= diff // 2
        bbox[2] += diff // 2
        bbox[0] = max(bbox[0], 0)
    elif diff < 0:
        bbox[1] += diff // 2
        bbox[3] -= diff // 2
        bbox[1] = max(bbox[1], 0)
    boxes.append(bbox[0])
    boxes.append(bbox[1])
    boxes.append(bbox[2])
    boxes.append(bbox[3])
    return boxes

def fix_box(img:np.array,bbox:list)->np.array:
    assert len(bbox) != 0, "bbox is empty"
    diff = (bbox[3] - bbox[1]) - (bbox[2] - bbox[0])
    if diff > 0:
        bbox[0] -= diff // 2
        bbox[2] += diff // 2
        bbox[0] = max(bbox[0], 0)
    elif diff < 0:
        bbox[1] += diff // 2
        bbox[3] -= diff // 2
        bbox[1] = max(bbox[1], 0)
    imgNew = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    H = bbox[3] - bbox[1]
    if H != FACESIZE:
        imgNew = cv2.resize(imgNew,(FACESIZE,FACESIZE),interpolation=cv2.INTER_CUBIC)
    return imgNew

def amplifyBox(img:np.array,boxes:list)->list and np.array:
    bbox = []
    assert len(boxes) != 0, "bbox is empty"
    h = (boxes[3] - boxes[1]) * FACEFACTOR
    w = (boxes[2] - boxes[0]) * FACEFACTOR
    x1 = boxes[0] - w // 2
    y1 = boxes[1] - h // 2
    x2 = boxes[2] + w // 2
    y2 = boxes[3] + h // 2

    H = (boxes[3] - boxes[1]) * (FACEFACTOR + 1)
    W = (boxes[2] - boxes[0]) * (FACEFACTOR + 1)
    imgTmp = np.full(shape=[H, W, 3], fill_value=255)

    if x1 < 0 and y1 > 0:
        x1 = 0
        img = img[y1:y2,x1:x2,:]
        dh = (imgTmp.shape[0] - img.shape[0]) // 2
        dw = (imgTmp.shape[1] - img.shape[1]) // 2
        nh = img.shape[0]
        nw = img.shape[1]
        imgTmp[dh:dh+nh,dw:dw+nw,:] = img
        bbox.append(x1)
        bbox.append(y1)
        bbox.append(x2)
        bbox.append(y2)

    elif x1 > 0 and y1 < 0:
        y1 = 0
        img = img[y1:y2,x1:x2,:]
        dh = (imgTmp.shape[0] - img.shape[0]) // 2
        dw = (imgTmp.shape[1] - img.shape[1]) // 2
        nh = img.shape[0]
        nw = img.shape[1]
        imgTmp[dh:dh + nh, dw:dw + nw, :] = img
        bbox.append(x1)
        bbox.append(y1)
        bbox.append(x2)
        bbox.append(y2)

    elif x1 < 0 and y1 < 0:
        x1 = 0
        y1 = 0
        img = img[y1:y2, x1:x2, :]
        dh = (imgTmp.shape[0] - img.shape[0]) // 2
        dw = (imgTmp.shape[1] - img.shape[1]) // 2
        nh = img.shape[0]
        nw = img.shape[1]
        imgTmp[dh:dh + nh, dw:dw + nw, :] = img
        bbox.append(x1)
        bbox.append(y1)
        bbox.append(x2)
        bbox.append(y2)

    else:
        imgTmp = img[y1:y2, x1:x2, :]
        bbox.append(x1)
        bbox.append(y1)
        bbox.append(x2)
        bbox.append(y2)

    return imgTmp,bbox

def crop(img:np.array, box:list)->np.array:
    # 放大比例 rate > 1
    rate = FACEFACTOR
    img_h, img_w, _ = img.shape

    # 按检测框直接剪裁的人脸
    # face_img = img[box[1]:box[3], box[0]:box[2], :]

    face_w = box[2] - box[0]
    face_h = box[3] - box[1]

    new_face_h = rate * face_h
    new_face_w = rate * face_w

    add_w = int((new_face_w - face_w) // 2)
    add_h = int((new_face_h - face_h) // 2)
    new_box = [box[0] - add_w, box[1] - add_h, box[2] + add_w, box[3] + add_h]

    # 剪裁人脸
    new_face_img = img[max(0, new_box[1]):min(img_h, new_box[3]), max(0, new_box[0]):min(img_w, new_box[2]), :]

    # 计算 上,下,左,右边补充像素数
    top = -min(new_box[1], 0)
    bottom = -min(img_h - new_box[3], 0)
    left = -min(new_box[0], 0)
    right = -min(img_w - new_box[2], 0)

    # 超出图片的区域填充像素
    new_face_img = cv2.copyMakeBorder(new_face_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255))

    # cv2.imshow("test", face_img)
    # cv2.imshow("pad", new_face_img)
    # cv2.waitKey()
    return new_face_img

def recoverImg(img:np.array):
    """
    :param img:
    :return:
    """
    H,W = img.shape[:2]
    dh0, dw0 = H // 5, W // 5
    imgNew = img[dh0:-dh0,dw0:-dw0,:]
    return imgNew

def recoverSquareImg(img:np.array):
    """
    :param img:
    :return:
    """
    H, W = img.shape[:2]
    x1 = integer(W / 5)
    y1 = integer(H / 5)
    x2 = 4 * x1
    y2 = 4 * y1
    H0 = y2 - y1
    W0 = x2 - x1
    diff = H0 - W0
    if diff > 0:
        x1 -= integer(diff/2)
        x2 += integer(diff/2)
        x1 = max(x1, 0)
    elif diff < 0:
        y1 += integer(diff/2)
        y2 -= integer(diff/2)
        y1 = max(y1, 0)
    imgNew = img[y1:y2, x1:x2, :]
    if imgNew.shape[0] != imgNew.shape[1]:
        size = max(imgNew.shape[0],imgNew.shape[1])
        imgNew = cv2.resize(imgNew,(size,size),interpolation=cv2.INTER_CUBIC)
    return imgNew

def alignFace(img:np.array,bounding_boxes:list,facial_landmarks:list)->np.array:
    """
    bounding_boxes = [x1,y1,x2,y2]
    facial_landmarks = [(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5)]
    """
    landMarks = []
    try:
        if bounding_boxes == [] or facial_landmarks == []:
            return []
        landMarks.append(facial_landmarks[0][0])
        landMarks.append(facial_landmarks[0][1])
        landMarks.append(facial_landmarks[1][0])
        landMarks.append(facial_landmarks[1][1])
        # print("landMarks=",landMarks)
        # print("bounding_boxes=",bounding_boxes)
        eye_center = ((landMarks[0] + landMarks[2]) // 2, (landMarks[1] + landMarks[3]) // 2)
        dy = landMarks[3] - landMarks[1]
        dx = landMarks[2] - landMarks[0]
        angle = cv2.fastAtan2(dy, dx)
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        img_rotation = cv2.warpAffine(img, M, dsize=(img.shape[1],img.shape[0]),borderValue=(255,255,255))

        x1 = bounding_boxes[0]
        y1 = bounding_boxes[1]
        x2 = bounding_boxes[2]
        y2 = bounding_boxes[3]

        imgSmall = img_rotation[y1:y2,x1:x2,:]
        img = cv2.resize(imgSmall,(FACESIZE,FACESIZE),interpolation=cv2.INTER_CUBIC)
        return img

    except Exception as e:
        print('unknown error {}'.format(e))