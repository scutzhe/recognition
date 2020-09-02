#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: split_occ_clean_file.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.11.14 11:44    shengyang      v0.1        creation

import os
import os.path as osp
import shutil


def move(input_dir, occ_output_dir, clean_output_dir):
    for root_dir, sub_dir, file_list in os.walk(input_dir):
        for f in file_list:
            if "_occ.jpg" in f:
                ori_file_path = osp.join(root_dir, f)
                tar_file_path = ori_file_path.replace(input_dir, occ_output_dir)
                tar_file_path.replace("_occ", "")
                tar_file_dir = osp.dirname(tar_file_path)
                if not osp.exists(tar_file_dir):
                    os.makedirs(tar_file_dir)
                shutil.move(ori_file_path, tar_file_path)
            elif ".jpg" in f:
                ori_file_path = osp.join(root_dir, f)
                tar_file_path = ori_file_path.replace(input_dir, clean_output_dir)
                tar_file_dir = osp.dirname(tar_file_path)
                if not osp.exists(tar_file_dir):
                    os.makedirs(tar_file_dir)
                shutil.move(ori_file_path, tar_file_path)
            else:
                raise Exception("{} file error".format(f))


if __name__ == "__main__":
    move(input_dir="/home/shengyang/haige_dataset/face_occusion/subset_for_test/imgs",
         occ_output_dir="/home/shengyang/haige_dataset/face_occusion/subset_for_test/occ_imgs",
         clean_output_dir="/home/shengyang/haige_dataset/face_occusion/subset_for_test/clean_imgs",
         )
