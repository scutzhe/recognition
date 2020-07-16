#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: megaface-split.py
# @Date: 19-8-14 下午8:49
# @Descr:

import csv
import os
import shutil
import os.path as osp
import json
import cv2
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
import tqdm


# with open("/home/shengyang/haige_dataset/face_dataset2/Trillion_Pairs/train_msra/msra_lmk", "rt") as f:
#     for i in range(10):
#         oneline = f.readline()
#         print(oneline)
#
# exit()


def process_one_subdir(root_dir, out_dir, pic_list):
    for f in pic_list:
        pic_path = osp.join(root_dir, f)
        json_path = pic_path + ".json"
        person_id = f.split("_")[0]
        tar_path = osp.join(out_dir, person_id, f)
        img = cv2.imread(pic_path)
        if not osp.exists(osp.dirname(tar_path)):
            os.makedirs(osp.dirname(tar_path))
        if not osp.exists(json_path):
            print("{} not exist".format(json_path))
            continue
        with open(json_path, 'rt') as fp:
            info = json.load(fp)
            bounding_box = info['bounding_box']
            x = int(bounding_box['x'])
            y = int(bounding_box['y'])
            w = int(bounding_box['width'])
            h = int(bounding_box['height'])
            face = img[y: y + h, x:x + w]
            cv2.imwrite(tar_path, face)


def split_by_person_id(megaface_dir="/home/shengyang/haige_dataset/face_dataset2/MegaFace/daniel/FlickrFinal2",
                       out_dir="/home/shengyang/haige_dataset/face_dataset2/mega"):
    executor = ThreadPoolExecutor(max_workers=16)
    for root_dir, sub_dir, filelist in tqdm.tqdm(os.walk(megaface_dir)):
        if sub_dir:
            continue
        pic_list = [f for f in filelist if f.endswith("jpg")]
        executor.submit(process_one_subdir, (root_dir, out_dir, pic_list))


def filter_by_person_num(megaface_dir="/home/shengyang/haige_dataset/face_dataset2/mega", n=4):
    count = 0
    for root_dir, sub_dir, filelist in os.walk(megaface_dir):
        pic_list = [f for f in filelist if f.endswith("jpg")]
        if len(pic_list) < n:
            count += 1
            # print(root_dir)
            # print(len(pic_list))
            # os.removedirs(root_dir)
    print(count)


if __name__ == "__main__":
    split_by_person_id()
