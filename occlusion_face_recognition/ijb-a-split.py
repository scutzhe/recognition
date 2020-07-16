#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: ijb-a-split.py
# @Date: 19-8-14 下午3:51
# @Descr:
import csv
import os
import shutil
import os.path as osp


content = []
ijb_a_train = "/media/ext/IJB-A/IJB-A_11_sets/split1/train_1.csv"
ijb_a_verify = "/media/ext/IJB-A/IJB-A_11_sets/split1/verify_metadata_1.csv"
for root_dir, _, filelist in os.walk("/media/ext/IJB-A/IJB-A_11_sets"):
    for f in filelist:
        if "train_1" in f or "verify_metadata_1" in f:
            f = osp.join(root_dir, f)
            with open(f, 'rt') as csvfile:
                csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
                next(csv_reader)  # 读取第一行每一列的标题
                for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
                    content.append(row)


file_dict = {row[2]: row[1] for row in content}

print(file_dict["img/8565.jpg"])

root_path = "/media/ext/IJB-A/CleanData"
out_dir = "/media/ext/IJB-A/Modify"
for root_dir, sub_dir, filelist in os.walk(root_path):
    for f in filelist:
        if f.endswith(".jpg"):
            abs_path = osp.join(root_dir, f)
            dir_name = abs_path[1+len(root_path):]
            person_id = file_dict[dir_name]
            tar_path = osp.join(out_dir, person_id, f)
            if not osp.exists(osp.dirname(tar_path)):
                os.makedirs(osp.dirname(tar_path))
            shutil.move(abs_path, tar_path)
