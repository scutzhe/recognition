#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : add_occlusion.py
# @time    : 8/17/20 9:44 AM
# @desc    : 
'''

import os
import cv2
from tqdm import tqdm
from faceDetection import faceDetectionCenterFace


if __name__ == "__main__":
    occlusion_face_dir = "/home/zhex/Downloads/mafa/images"
    save_dir = "/home/zhex/Downloads/occlusion_face"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_names = os.listdir(occlusion_face_dir)
    index = 1
    for name in tqdm(image_names):
        image_path = os.path.join(occlusion_face_dir,name)
        try:
            image = cv2.imread(image_path)
            img, _ = faceDetectionCenterFace(image)
            index += 1
            # cv2.imshow("image",img)
            # cv2.waitKey(1)
            cv2.imwrite(save_dir + "/" + "{}.png".format(index),img)
        except Exception as e:
            print(e)

