#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : image_test.py
# @time    : 8/26/20 2:58 PM
# @desc    : 
'''

from PIL import Image

image_path = "testImage/4.png"
image = Image.open(image_path)
W,H = image.size
print("W,H=",W,H)
image.show()
image.thumbnail((W//2,H//2))
image.save("test.png","jpeg")