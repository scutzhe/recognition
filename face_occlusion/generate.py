#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author  : 郑祥忠
@license : (C) Copyright,2013-2019,广州海格星航科技
@contact : dylenzheng@gmail.com
@file    : demo.py
@time    : 9/6/19 5:17 PM
@desc    :
'''
import cv2
import os
import numpy as np
from PIL import Image
from img_occ import occ
from detector import detect_faces
from visualization_utils import rotation_img,show_results
import time
import random

def face_detection(img):
    bounding_boxes, landmarks = detect_faces(img)
    img_set = rotation_img(img, bounding_boxes, landmarks)
    #print('bounding_boxes = ',bounding_boxes)
    return img_set,landmarks,bounding_boxes


def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              #print(path)
              _files.append(path)
    return _files


def generate_data(dataset_path, image_path, mask_path):
    """
    产生数据
    dataset_path: 数据集路径
    image_path: 输出图片保存路径
    mask_path:  输出mask保存路径
    mod_list = [3, 7, 8] 即当前mod的模式：occ3,occ7,occ8
    """
    dirmenu = os.listdir(dataset_path)
    mod_list = [3, 7, 8]
    for i, sub_dir in enumerate(dirmenu, 0):
        person_path = sub_dir
        sub_dir = os.path.join(dataset_path, sub_dir)
        img_list = os.listdir(sub_dir)
        for img_name in img_list:
            img_path = os.path.join(sub_dir, img_name)
            # print("img_path", img_path)
            img = Image.open(img_path)
            img_set, landmarks, bounding_boxes = face_detection(img)
            i = 0
            for img, land, boxes in zip(img_set, landmarks, bounding_boxes):
                i = i + 1
                if i == 1:
                    y1 = int(land[0] - boxes[0] + 5)  # 左眼
                    x1 = int(land[5] - boxes[1] + 5)
                    y2 = int(land[1] - boxes[0] + 5)  # 右眼
                    x2 = int(land[6] - boxes[1] + 5)
                    y3 = int(land[2] - boxes[0] + 5)  # 鼻子
                    x3 = int(land[7] - boxes[1] + 5)
                    keypoint = np.asarray([x1, y1, x2, y2, x3, y3])
                    random_mod = random.randint(0, len(mod_list)-1)
                    img_binary, img_occ, boxes_occ, mod = occ(img, keypoint, mod=mod_list[random_mod])

                    pic_path = os.path.join(image_path+"/image", person_path)
                    if not os.path.exists(pic_path):
                        os.makedirs(pic_path)

                    mask_dir = os.path.join(mask_path+"/mask", person_path)
                    if not os.path.exists(mask_dir):
                        os.makedirs(mask_dir)

                    temp_img = os.path.join(pic_path, img_name)
                    temp_mask = os.path.join(mask_dir, img_name)

                    cv2.imwrite(temp_img, img_occ)
                    cv2.imwrite(temp_mask, img_binary)

# start = time.time()
# generate_data("/home/chenwenhao/face-10w/CASIA-WebFace_part","/home/chenwenhao/face-10w", "/home/chenwenhao/face-10w")
# print("time:{}".format(time.time()-start))