#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : testInputSize.py
# @time    : 7/24/20 8:59 AM
# @desc    : 
'''

import os
from PIL import Image
from backbone.selfdefine import CaffeCrop

caffeCropTest = CaffeCrop("test")

def checkSize(image,mode):
    """
    @param image:
    @param mode:
    @return:
    """
    caffeCrop = CaffeCrop(mode)
    newImg = caffeCrop.__call__(image)
    return newImg

if __name__ == "__main__":
    imgDir = "testImage"
    for imgName in os.listdir(imgDir):
        imgPath = os.path.join(imgDir,imgName)
        img = Image.open(imgPath)
        print("img.size=",img.size)
        newImg = checkSize(img,"test")
        print("newImg.size=",newImg.size)


