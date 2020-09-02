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
file_path = '/home/wuzhouping/lfw/lfw'
print(file_path)
#cut_file_path = 'ms1m_cut/'
occ_file_path = 'imgs_occ/'
binary_file_path = 'imgs_binary/'
#txt = open('../binary.txt', 'a')
begintime = time.time()
print ('bengin!')
files = list_all_files(file_path)
print ('load path!')
file_len = len(files)
print ('file_len = ',file_len)
n = 0
for f in files:
    n = n+1
    print(n/file_len)
    file_name =f.rstrip().split('/')[-2]
    img_id = f.rstrip().split('/')[-1]
    #cut_path = os.path.join(cut_file_path, file_name)
    occ_path = os.path.join(occ_file_path, file_name)
    binary_path = os.path.join(binary_file_path, file_name)
    #if not os.path.exists(cut_path):
    #    os.makedirs(cut_path)
    #if not os.path.exists(occ_path):
    #    os.makedirs(occ_path)
    #if not os.path.exists(binary_path):
        #os.makedirs(binary_path)
    id = img_id.rstrip().split('.')[-2]
    #print (occ_path)

    #img_cut_save = os.path.join(occ_path,'/',img_id)

    #img_occ_save = os.path.join(occ_path, '/', id, '_occ.jpg')
    #img_bin_save = os.path.join(binary_path, '/', id, '_occ.jpg')
    img_cut_save = occ_path + '/' +img_id
    img_occ_save = occ_path + '/' + id + '_occ.jpg'
    img_bin_save = binary_path + '/' + id + '_occ.jpg'
    #print (img_cut_save)

    img = Image.open(f)
    img_set,landmarks,bounding_boxes = face_detection(img)

    # img_copy = show_results(img, bounding_boxes, landmarks)
    # img_copy.save('22.jpg')
    # img = np.asarray(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    i = 0
    for img, land, boxes in zip(img_set,landmarks,bounding_boxes):
        i = i+1
        if i == 1:
            y1 = int(land[0] - boxes[0] + 5)        #左眼
            x1 = int(land[5] - boxes[1] + 5)
            y2 = int(land[1] - boxes[0] + 5)        #右眼
            x2 = int(land[6] - boxes[1] + 5)
            y3 = int(land[2] - boxes[0] + 5)        #鼻子
            x3 = int(land[7] - boxes[1] + 5)
            keypoint = np.asarray([x1,y1,x2,y2,x3,y3])
            #print ('keypoint = ',keypoint)
            cv2.imwrite(img_cut_save,img)
            img_binary, img_occ, boxes_occ, mod = occ(img, keypoint, mod = 6)
            if mod == 1:
                #print('1111')
                file_path_occ = '../1/'+occ_path
                file_path_bin = '../1/'+binary_path
                if not os.path.exists(file_path_occ):
                    os.makedirs(file_path_occ)
                if not os.path.exists(file_path_bin):
                    os.makedirs(file_path_bin)
                cv2.imwrite('../1/'+img_occ_save,img_occ)
                # cv2.imwrite('../1/'+img_bin_save,img_binary)
            elif mod == 2:
                file_path_occ = '../2/'+occ_path
                file_path_bin = '../2/'+binary_path
                if not os.path.exists(file_path_occ):
                    os.makedirs(file_path_occ)
                if not os.path.exists(file_path_bin):
                    os.makedirs(file_path_bin)
                cv2.imwrite('../2/'+img_occ_save,img_occ)
                # cv2.imwrite('../2/'+img_bin_save,img_binary)
            elif mod == 3 or mod == 10:
                file_path_occ = '../3/'+occ_path
                file_path_bin = '../3/'+binary_path
                if not os.path.exists(file_path_occ):
                    os.makedirs(file_path_occ)
                if not os.path.exists(file_path_bin):
                    os.makedirs(file_path_bin)
                cv2.imwrite('../3/'+img_occ_save,img_occ)
                # cv2.imwrite('../3/'+img_bin_save,img_binary)
            elif mod == 4:
                file_path_occ = '../4/'+occ_path
                file_path_bin = '../4/'+binary_path
                if not os.path.exists(file_path_occ):
                    os.makedirs(file_path_occ)
                if not os.path.exists(file_path_bin):
                    os.makedirs(file_path_bin)
                cv2.imwrite('../4/'+img_occ_save,img_occ)
                # cv2.imwrite('../4/'+img_bin_save,img_binary)
            elif mod == 5:
                file_path_occ = '../5/'+occ_path
                file_path_bin = '../5/'+binary_path
                if not os.path.exists(file_path_occ):
                    os.makedirs(file_path_occ)
                if not os.path.exists(file_path_bin):
                    os.makedirs(file_path_bin)
                cv2.imwrite('../5/'+img_occ_save,img_occ)
                # cv2.imwrite('../5/'+img_bin_save,img_binary)
            elif mod == 6:
                file_path_occ = '../6/'+occ_path
                file_path_bin = '../6/'+binary_path
                if not os.path.exists(file_path_occ):
                    os.makedirs(file_path_occ)
                if not os.path.exists(file_path_bin):
                    os.makedirs(file_path_bin)
                cv2.imwrite('../6/'+img_occ_save,img_occ)
                # cv2.imwrite('../6/'+img_bin_save,img_binary)
            elif mod == 7 or mod == 9:
                file_path_occ = '../7/'+occ_path
                file_path_bin = '../7/'+binary_path
                if not os.path.exists(file_path_occ):
                    os.makedirs(file_path_occ)
                if not os.path.exists(file_path_bin):
                    os.makedirs(file_path_bin)
                cv2.imwrite('../7/'+img_occ_save,img_occ)
                # cv2.imwrite('../7/'+img_bin_save,img_binary)
            elif mod == 8 :
                #print('8888')
                file_path_occ = '../8/'+occ_path
                file_path_bin = '../8/'+binary_path
                if not os.path.exists(file_path_occ):
                    os.makedirs(file_path_occ)
                if not os.path.exists(file_path_bin):
                    os.makedirs(file_path_bin)
                cv2.imwrite('../8/'+img_occ_save,img_occ)
                # cv2.imwrite('../8/'+img_bin_save,img_binary)
            else:
                file_path_occ = '../full/'+occ_path
                file_path_bin = '../full/'+binary_path
                if not os.path.exists(file_path_occ):
                    os.makedirs(file_path_occ)
                if not os.path.exists(file_path_bin):
                    os.makedirs(file_path_bin)
                cv2.imwrite('../full/'+img_occ_save,img_occ)
                # cv2.imwrite('../full/'+img_bin_save,img_binary)
                # print('error!!!')
            
            
            #print ('boxes_occ = ',boxes_occ)
            #cla = str(boxes_occ[0])
            #xx1 = str(boxes_occ[1])
            #yy1 = str(boxes_occ[2])
            #xx2 = str(boxes_occ[3])
            #yy2 = str(boxes_occ[4])
            #xx3 = str(boxes_occ[5])
            #yy3 = str(boxes_occ[6])
            #txt.write(img_occ_save+' '+cla+ ' '+xx1+' ' +yy1+' '+xx2+' '+yy2+' '+xx3+' '+yy3+'\n')
        else:
            break

print ('flish ！')
endtime = time.time()
costtime = endtime-begintime
print('costtime = ',costtime)
