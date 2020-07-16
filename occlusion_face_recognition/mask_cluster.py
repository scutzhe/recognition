#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: mask_cluster.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.11.15 14:25    shengyang      v0.1        creation

import numpy as np
import cv2
from sklearn.cluster import k_means
import os
import os.path as osp
import pickle
import threading


mask_file_pkl = "mask_file_list.pkl"
mask_feature_pkl = "mask_features.pkl"
total_num = 500000

mask_feature_list = [None] * total_num


def read_and_getfeatures(file_path_list, start):
    for i, f in enumerate(file_path_list):
        gray = cv2.imread(f)[:, :, 1]
        mask = cv2.resize(gray, (7, 7))
        _, mask = cv2.threshold(mask, 128, 1, 0)
        fea = mask.reshape((-1))
        mask_feature_list[i + start] = fea


def get_all_mask(dirpath):
    if osp.isfile(mask_feature_pkl):
        print("found mask_feature_pkl, reading")
        with open(mask_feature_pkl, 'rb') as f:
            mask_features = pickle.load(f)
        return mask_features

    if osp.isfile(mask_file_pkl):
        with open("mask_file_list.pkl", 'rb') as f:
            mask_file_list = pickle.load(f)
        print("in pkl file found files:", len(mask_file_list))
    else:
        mask_file_list = []
        for root_dir, sub_dir, file_list in os.walk(dirpath):
            abs_file_list = [osp.join(root_dir, f) for f in file_list if f.endswith(".jpg")]
            mask_file_list.extend(abs_file_list)
            if len(mask_file_list) > total_num:
                break
        print("found files:", len(mask_file_list))
        mask_file_list = mask_file_list[: total_num]
        with open("mask_file_list.pkl", 'wb') as f:
            pickle.dump(mask_file_list, f)

    tid = []
    for i in range(10):
        batch = total_num // 10
        start = i * batch
        t = threading.Thread(target=read_and_getfeatures,
                             args=(mask_file_list[start: start+batch], start))
        t.start()
        tid.append(t)

    for t in tid:
        t.join()

    mask_features = np.array(mask_feature_list)
    with open(mask_feature_pkl, 'wb') as f:
        pickle.dump(mask_features, f)

    return mask_features


if __name__ == "__main__":
    features = get_all_mask("/home/shengyang/haige_dataset/face_occusion/real_occlusion2/mask")
    # print(features.shape)
    best_centers, best_labels, best_inertia = k_means(X=features, n_clusters=20)
    _, best_centers = cv2.threshold(best_centers, 0.5, 1, 0)
    best_centers = best_centers.reshape((-1, 7, 7))
    print(best_centers)
