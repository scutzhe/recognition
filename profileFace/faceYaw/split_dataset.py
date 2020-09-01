#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : split_dataset.py
# @time    : 8/16/20 8:44 AM
# @desc    : 
'''
import os
import shutil
import random
from tqdm import  tqdm


def move_file(src_dir, target_dir):
    """
    :param src_dir:
    :param target_dir:
    :return:
    """
    assert os.path.exists(src_dir),"{} is null !!!".format(src_dir)
    assert os.path.exists(target_dir),"{} is null !!!".format(target_dir)

    image_names = os.listdir(src_dir)
    for name in tqdm(image_names):
        image_path = os.path.join(src_dir,name)
        shutil.move(image_path,target_dir)

def add_profile(src_dir, target_dir):
    """
    :param src_dir:
    :param target_dir:
    :return:
    """
    assert os.path.exists(src_dir), "{} is null !!!".format(src_dir)
    assert os.path.exists(target_dir), "{} is null !!!".format(target_dir)
    image_names = os.listdir(src_dir)
    # names = random.sample(image_names,35000)
    names = image_names
    for name in tqdm(names):
        image_path = os.path.join(src_dir,name)
        shutil.move(image_path, target_dir)

def add_frontal(src_dir, target_dir):
    """
    :param src_dir:
    :param target_dir:
    :return:
    """
    assert os.path.exists(src_dir), "{} is null !!!".format(src_dir)
    assert os.path.exists(target_dir), "{} is null !!!".format(target_dir)
    image_names = os.listdir(src_dir)
    names = random.sample(image_names,123909)
    for name in tqdm(names):
        image_path = os.path.join(src_dir,name)
        shutil.copy(image_path, target_dir)

def frontal_yaw(origin_train_frontal_dir,origin_test_frontal_dir,new_train_frontal_dir,new_test_frontal_dir):
    """
    :param origin_train_frontal_dir:
    :param origin_test_frontal_dir:
    :param new_train_frontal_dir:
    :param new_test_frontal_dir:
    :return:
    """
    assert os.path.exists(origin_train_frontal_dir),"{} is null !!!".format(origin_train_frontal_dir)
    assert os.path.exists(origin_test_frontal_dir),"{} is null !!!".format(origin_test_frontal_dir)
    if not os.path.exists(new_train_frontal_dir):
        os.makedirs(new_train_frontal_dir)
    if not os.path.exists(new_test_frontal_dir):
        os.makedirs(new_test_frontal_dir)

    origin_train_frontal_names = os.listdir(origin_train_frontal_dir)
    origin_test_frontal_names = os.listdir(origin_test_frontal_dir)
    index_train = 0
    index_test = 0
    for name in tqdm(origin_train_frontal_names):
        tmp = name.split(".")[0]
        yaw = int(str(tmp).split("_")[1])
        # print("yaw=",yaw)
        image_path = os.path.join(origin_train_frontal_dir,name)
        if yaw <= 15:##定义新的正脸角度小于15°
            shutil.copy(image_path,new_train_frontal_dir)
            index_train += 1
        else:
            pass
    for name in tqdm(origin_test_frontal_names):
        tmp = name.split(".")[0]
        yaw = int(str(tmp).split("_")[1])
        # print("yaw=",yaw)
        image_path = os.path.join(origin_test_frontal_dir,name)
        if yaw <= 15:##定义新的正脸角度小于15°
            shutil.copy(image_path,new_test_frontal_dir)
            index_test += 1
        else:
            pass
    print("index_train=",index_train)
    print("index_test=",index_test)

def profile_yaw(origin_train_profile_dir,origin_test_profile_dir,new_train_profile_dir,new_test_profile_dir):
    """
    :param origin_train_profile_dir:
    :param origin_test_profile_dir:
    :param new_train_profile_dir:
    :param new_test_profile_dir:
    :return:
    """
    assert os.path.exists(origin_train_profile_dir),"{} is null !!!".format(origin_train_profile_dir)
    assert os.path.exists(origin_test_profile_dir),"{} is null !!!".format(origin_test_profile_dir)
    if not os.path.exists(new_train_profile_dir):
        os.makedirs(new_train_profile_dir)
    if not os.path.exists(new_test_profile_dir):
        os.makedirs(new_test_profile_dir)

    origin_train_profile_names = os.listdir(origin_train_profile_dir)
    origin_test_profile_names = os.listdir(origin_test_profile_dir)
    index_train = 0
    index_test = 0
    for name in tqdm(origin_train_profile_names):
        tmp = name.split(".")[0]
        yaw = int(str(tmp).split("_")[1])
        # print("yaw=",yaw)
        image_path = os.path.join(origin_train_profile_dir,name)
        if yaw <= 15:##定义新的正脸角度小于15°
            shutil.copy(image_path,new_train_profile_dir)
            index_train += 1
        else:
            pass
    for name in tqdm(origin_test_profile_names):
        tmp = name.split(".")[0]
        yaw = int(str(tmp).split("_")[1])
        # print("yaw=",yaw)
        image_path = os.path.join(origin_test_profile_dir,name)
        if yaw <= 15:##定义新的正脸角度小于15°
            shutil.copy(image_path,new_test_profile_dir)
            index_test += 1
        else:
            pass
    print("index_train=",index_train)
    print("index_test=",index_test)


def equal_dataset(train_frontal_dir,train_profile_dir,middle_dir):
    """
    :param train_frontal_dir:
    :param train_profile_dir:
    :param middle_dir:
    :return:
    """
    assert os.path.exists(train_frontal_dir), "{} is null".format(train_frontal_dir)
    assert os.path.exists(train_profile_dir), "{} is null".format(train_profile_dir)
    if not os.path.exists(middle_dir):
        os.makedirs(middle_dir)
    frontal_names = os.listdir(train_frontal_dir)
    profile_names = os.listdir(train_profile_dir)
    frontal_length = len(frontal_names)
    profile_length = len(profile_names)
    length = profile_length if profile_length < frontal_length else frontal_length
    if length == profile_length:
        middle_names = random.sample(frontal_names,length)
        for name in tqdm(middle_names):
            image_path = os.path.join(train_frontal_dir,name)
            shutil.copy(image_path,middle_dir)
    else:
        middle_names = random.sample(profile_names,length)
        for name in tqdm(middle_names):
            image_path = os.path.join(train_profile_dir,name)
            shutil.copy(image_path,middle_dir)

def split_dataset(train_frontal_dir,train_profile_dir,test_frontal_dir,test_profile_dir):
    """
    :param train_frontal_dir:
    :param train_profile_dir:
    :param test_frontal_dir:
    :param test_profile_dir:
    :return:
    """
    assert os.path.exists(train_frontal_dir),"{} is null".format(train_frontal_dir)
    assert os.path.exists(train_profile_dir),"{} is null".format(train_profile_dir)

    if not os.path.exists(test_frontal_dir):
        os.makedirs(test_frontal_dir)
    if not os.path.exists(test_profile_dir):
        os.makedirs(test_profile_dir)

    frontal_names = os.listdir(train_frontal_dir)
    profile_names = os.listdir(train_profile_dir)
    train_length = len(frontal_names)
    test_length = int(0.2 * train_length)
    print("test_length=",test_length)

    indexF = 0
    for name in tqdm(frontal_names):
        image_path = os.path.join(train_frontal_dir,name)
        shutil.move(image_path,test_frontal_dir)
        indexF += 1
        if indexF == test_length:
            break

    indexP = 0
    for name in tqdm(profile_names):
        image_path = os.path.join(train_profile_dir, name)
        shutil.move(image_path, test_profile_dir)
        indexP += 1
        if indexP == test_length:
            break

    print("indexF,indexP = ",indexF, indexP)

def get_sub_dataset(origin_train_frontal_dir,origin_train_profile_dir,origin_test_frontal_dir,origin_test_profile_dir,
                    subset_train_frontal_dir,subset_train_profile_dir,subset_test_frontal_dir,subset_test_profile_dir):
    """
    :param origin_train_frontal_dir:
    :param origin_train_profile_dir:
    :param origin_test_frontal_dir:
    :param origin_test_profile_dir:
    :param subset_train_frontal_dir:
    :param subset_train_profile_dir:
    :param subset_test_frontal_dir:
    :param subset_test_profile_dir:
    :return:
    """
    assert os.path.exists(origin_train_frontal_dir),"{} is null !!!".format(origin_train_frontal_dir)
    assert os.path.exists(origin_train_profile_dir),"{} is null !!!".format(origin_train_profile_dir)
    assert os.path.exists(origin_test_frontal_dir),"{} is null !!!".format(origin_test_frontal_dir)
    assert os.path.exists(origin_test_profile_dir),"{} is null !!!".format(origin_test_profile_dir)

    if not os.path.exists(subset_train_frontal_dir):
        os.makedirs(subset_train_frontal_dir)
    if not os.path.exists(subset_train_profile_dir):
        os.makedirs(subset_train_profile_dir)
    if not os.path.exists(subset_test_frontal_dir):
        os.makedirs(subset_test_frontal_dir)
    if not os.path.exists(subset_test_profile_dir):
        os.makedirs(subset_test_profile_dir)
    origin_train_frontal_names = os.listdir(origin_train_frontal_dir)
    origin_train_profile_names = os.listdir(origin_train_profile_dir)
    origin_test_frontal_names = os.listdir(origin_test_frontal_dir)
    origin_test_profile_names = os.listdir(origin_test_profile_dir)

    o_train_f_names = random.sample(origin_train_frontal_names,20000)
    o_train_p_names = random.sample(origin_train_profile_names,20000)
    o_test_f_names = random.sample(origin_test_frontal_names,4000)
    o_test_p_names = random.sample(origin_test_profile_names,4000)

    for name in tqdm(o_train_f_names):
        image_path = os.path.join(origin_train_frontal_dir,name)
        shutil.copy(image_path,subset_train_frontal_dir)
    for name in tqdm(o_train_p_names):
        image_path = os.path.join(origin_train_profile_dir,name)
        shutil.copy(image_path,subset_train_profile_dir)
    for name in tqdm(o_test_f_names):
        image_path = os.path.join(origin_test_frontal_dir,name)
        shutil.copy(image_path,subset_test_frontal_dir)
    for name in tqdm(o_test_p_names):
        image_path = os.path.join(origin_test_profile_dir,name)
        shutil.copy(image_path,subset_test_profile_dir)

def split_frontal_profile(origin_image_dir,train_frontal_image_dir,train_profile_image_dir):
    """
    :param origin_image_dir:
    :param train_frontal_image_dir:
    :param train_profile_image_dir:
    :return:
    """
    assert os.path.exists(origin_image_dir),"{} is null !!!".format(origin_image_dir)
    if not os.path.exists(train_frontal_image_dir):
        os.makedirs(train_frontal_image_dir)
    if not os.path.exists(train_profile_image_dir):
        os.makedirs(train_profile_image_dir)

    image_names = os.listdir(origin_image_dir)
    indexF_train =0
    indexP_train = 0
    indexF_test = 0
    indexP_test = 0
    for name in tqdm(image_names):
        image_path = os.path.join(origin_image_dir,name)
        flag = int(str(name.split(".")[0]).split("_")[1])
        if flag <= 5:
            indexF_train += 1
            shutil.copy(image_path,train_frontal_image_dir)
        elif flag >= 45:
            indexP_train += 1
            shutil.copy(image_path,train_profile_image_dir)
        else:
            pass

    print("indexF_train=",indexF_train)
    print("indexP_train=",indexP_train)

## step 1
# if __name__ == "__main__":
#     frontal_dir = "/home/zhex/Downloads/face_classification_full/train/frontal_origin"
#     profile_dir = "/home/zhex/Downloads/face_classification_full/train/profile"
#     middle_dir = "/home/zhex/Downloads/face_classification_full/train/frontal"
#     equal_dataset(frontal_dir,profile_dir,middle_dir)


## step 2
# if __name__ == "__main__":
#     train_frontal_dir = "/home/zhex/Downloads/face_classification_full/train/frontal"
#     train_profile_dir = "/home/zhex/Downloads/face_classification_full/train/profile"
#
#     test_frontal_dir = "/home/zhex/Downloads/face_classification_full/test/frontal"
#     test_profile_dir = "/home/zhex/Downloads/face_classification_full/test/profile"
#     split_dataset(train_frontal_dir, train_profile_dir, test_frontal_dir, test_profile_dir)

# if __name__ == "__main__":
#     origin_train_frontal_dir = "/home/zhex/Downloads/face_classification_full_2030_12/train/frontal"
#     origin_test_frontal_dir = "/home/zhex/Downloads/face_classification_full_2030_12/test/frontal"
#
#     new_train_frontal_dir = "/home/zhex/Downloads/face_classification_full_1530_12/train/frontal"
#     new_test_frontal_dir = "/home/zhex/Downloads/face_classification_full_1530_12/test/frontal"
#     frontal_yaw(origin_train_frontal_dir, origin_test_frontal_dir, new_train_frontal_dir, new_test_frontal_dir)

# if __name__ == "__main__":
#     # src_dir = "/home/zhex/Downloads/face_classification_full/test/frontal"
#     # target_dir = "/home/zhex/Downloads/face_classification_full/train/frontal"
#     # move_file(src_dir,target_dir)
#     # src_dir = "/home/zhex/Downloads/middle (copy)"
#     # target_dir = "/home/zhex/Downloads/face_classification_full/train/profile"
#     # add_profile(src_dir,target_dir)
#     src_dir = "/home/zhex/Downloads/frontal_origin"
#     target_dir = "/home/zhex/Downloads/face_classification_full/train/frontal"
#     add_frontal(src_dir,target_dir)


# if __name__ == "__main__":
#     origin_train_frontal_dir = "/home/zhex/Downloads/face_classification_full_1530_12/train/frontal"
#     origin_train_profile_dir = "/home/zhex/Downloads/face_classification_full_1530_12/train/profile"
#     origin_test_frontal_dir = "/home/zhex/Downloads/face_classification_full_1530_12/test/frontal"
#     origin_test_profile_dir = "/home/zhex/Downloads/face_classification_full_1530_12/test/profile"
#     subset_train_frontal_dir = "/home/zhex/Downloads/face_classification_full_1530_2/train/frontal"
#     subset_train_profile_dir = "/home/zhex/Downloads/face_classification_full_1530_2/train/profile"
#     subset_test_frontal_dir = "/home/zhex/Downloads/face_classification_full_1530_2/test/frontal"
#     subset_test_profile_dir = "/home/zhex/Downloads/face_classification_full_1530_2/test/profile"
#     get_sub_dataset(origin_train_frontal_dir, origin_train_profile_dir, origin_test_frontal_dir,
#                     origin_test_profile_dir,subset_train_frontal_dir, subset_train_profile_dir,
#                     subset_test_frontal_dir,subset_test_profile_dir)

# if __name__ == "__main__":
#     origin_image_dir = "/home/zhex/data/face_classification_full/images"
#     train_frontal_image_dir = "/home/zhex/data/face_classification_0545/train/frontal"
#     train_profile_image_dir = "/home/zhex/data/face_classification_0545/train/profile"
#     split_frontal_profile(origin_image_dir, train_frontal_image_dir, train_profile_image_dir)