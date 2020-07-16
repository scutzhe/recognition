#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2013-2019,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : rename.py
# @time    : 11/18/19 11:13 AM
# @desc    : 
'''

# import os
# import shutil
# dir1 = '/home/zhex/test_result/recognition/gallery_probe_dream/gallery'
# dir2 = '/home/zhex/test_result/recognition/gallery_probe_dream/probe'
#
# file_gallery = open('probe.txt','a')
# for root, dirs, files in os.walk(dir2):
#     for img_name in files:
#         new_name = 'gallery_probe_dream/probe/'+img_name
#         file_gallery.write(new_name+'\n')

# for root, dirs ,files in os.walk(dir1):
#     for img_name in files:
#         img_path = os.path.join(root,img_name)
#         img_new = img_path.replace('/','_').replace('-','_')
#         img_name_new = img_new.split('_',10)[10]
#
#         new_name = img_name.replace(img_name,img_name_new)
#         os.rename(img_path,os.path.join(root,new_name))
#         shutil.copy(os.path.join(root,new_name),dir2)
#         print('new_name =',new_name)