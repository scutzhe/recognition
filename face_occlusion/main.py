#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: test.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.10.22 08:57    samon      v0.1        creation

import cv2
import os
import os.path as osp
import numpy as np
from PIL import Image
from img_occ import face_occlusion
from face_detection.detection import Facedetection
from face_detection.crop_and_align import Cropper
from visualization_utils import rotation_img, show_results

import multiprocessing
import time
import sys


def face_detection(dector, img):
    """
    detect face from a image, and return aligned images, landmarks, bounding_boxes
    :param img:
    :return:
    """
    bounding_boxes, landmarks = dector.detect_faces(img, min_face_size=20,
                                                    thresholds=(0.4, 0.5, 0.6),
                                                    nms_thresholds=(0.5, 0.5, 0.5))

    aligned_imgs = rotation_img(img, bounding_boxes, landmarks)
    if len(bounding_boxes) is 0:
        cv2.imshow("test", np.asarray(img))
        cv2.waitKey(0)
    else:
        print(bounding_boxes)

    return aligned_imgs, landmarks, bounding_boxes


def face_detection_centerface(dector, img_rgb, keep_one=True):
    """
    detect face from a image, and return aligned images, landmarks, bounding_boxes
    :param img_rgb:
    :return:
    """
    bounding_boxes, scores, landmarks = dector.face_detect([img_rgb], debug=True)
    bounding_boxes = bounding_boxes[0]
    scores = scores[0]
    landmarks = landmarks[0]

    # 默认使用第0个检测目标
    landmark = landmarks[0].reshape(-1)
    bounding_box = bounding_boxes[0]

    if keep_one and len(bounding_boxes) > 1:
        img_w, img_h, _ = img_rgb.shape
        face_area_list = []
        center_dis_list = []
        for one_box in bounding_boxes:
            face_w = one_box[2] - one_box[0]
            face_h = one_box[3] - one_box[1]
            face_area_list.append(face_w * face_h)
            center_dis_list.append(((one_box[2] + one_box[0]) / 2 - img_w) * ((one_box[3] + one_box[1]) / 2 - img_h))
        bigest_index = np.asarray(face_area_list).argmax()
        most_central = np.asarray(center_dis_list).argmin()
        if bigest_index == most_central:
            bounding_box = bounding_boxes[0][bigest_index]
            landmark = landmarks[bigest_index].reshape(-1)

    # face_list, align_face_list, new_landmarks = Cropper().crop(img_rgb, bounding_boxes, landmarks)

    return img_rgb, landmark.reshape(-1), bounding_box


def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            # print(path)
            _files.append(path)
    return _files


def do_by_filelist(dector, filelist, occlusion_dir, mask_dir, aligned_dir, split_by_mode=False):
    import random
    random.shuffle(filelist)
    for file_path in filelist:
        file_name = osp.basename(file_path)[:-4]    # just file name, not contain file suffix
        file_suffix = file_path[-4:]
        person_id = osp.basename(osp.dirname(file_path))

        # img = Image.open(file_path)
        img_rgb = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        # st = time.time()
        faces_rgb, landmarks, bounding_boxes = face_detection_centerface(dector, img_rgb)
        # print("{}".format(time.time()-st))
        if len(landmarks) is 0:
            print(file_path)
            continue

        face_rgb = faces_rgb
        land = landmarks.reshape(-1)

        box = bounding_boxes
        # left_eye_y = int(land[0] - box[0])  # left eye
        # left_eye_x = int(land[5] - box[1])
        # right_eye_y = int(land[1] - box[0])  # right eye
        # right_eye_x = int(land[6] - box[1])
        # nose_y = int(land[2] - box[0])  # nose
        # nose_x = int(land[7] - box[1])
        # keypoint = np.asarray([left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y])

        # [l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y, mouth_l_x, mouth_l_y, mouth_r_x, mouth_r_y]
        keypoint = land

        img_mask, img_occlusion, boxes_occlusion, occlusion_mod = face_occlusion(face_rgb, keypoint)
        # print(img_occlusion.shape)
        # print(img_mask.shape)
        # occlusion_mod == 1, 以鼻尖为起点,图片四角(随机一个角)为终点的矩形遮挡
        # occlusion_mod == 2, 以鼻尖竖线为起始边,图片左右两边(随机一边)为终边的矩形遮挡
        # occlusion_mod == 3, 口罩贴图遮挡
        # occlusion_mod == 4, 遮挡眼睛鼻子之外部分
        # occlusion_mod == 5, 遮挡双眼和鼻子的三角区域
        # occlusion_mod == 6, 遮挡除眼睛之外的部分
        # occlusion_mod == 7, 墨镜贴图遮挡
        # occlusion_mod == 8, 帽子贴图遮挡
        # else, 不遮挡

        if split_by_mode is True:
            occlusion_path = osp.join(occlusion_dir, str(occlusion_mod), person_id, file_name+"_occ") + file_suffix
            mask_path = osp.join(mask_dir, str(occlusion_mod), person_id, file_name+"_occ") + file_suffix
            # aligned_path = osp.join(aligned_dir, str(occlusion_mod), person_id, file_name) + file_suffix
        else:
            occlusion_path = osp.join(occlusion_dir, person_id, file_name+"_occ") + file_suffix
            mask_path = osp.join(mask_dir, person_id, file_name+"_occ") + file_suffix
            aligned_path = osp.join(occlusion_dir, person_id, file_name) + file_suffix
            # aligned_path = osp.join(aligned_dir, person_id, file_name) + file_suffix

        if not osp.exists(osp.dirname(occlusion_path)):
            os.makedirs(osp.dirname(occlusion_path))
        if not osp.exists(osp.dirname(mask_path)):
            os.makedirs(osp.dirname(mask_path))
        # if not osp.exists(osp.dirname(aligned_path)):
        #     os.makedirs(osp.dirname(aligned_path))

        # cv2.imwrite(aligned_path, img_aligned)
        img_bgr_occlusion = cv2.cvtColor(img_occlusion, cv2.COLOR_RGB2BGR)
        cv2.imwrite(occlusion_path, img_bgr_occlusion)
        cv2.imwrite(mask_path, img_mask)


def start_occlusion(input_dir="resource/test_images", output_dir="output"):
    """
    start occlusion
    :param input_dir: input image file dir
    :param output_dir: output image dir
    :return:
    """
    occlusion_dir = osp.join(output_dir, 'imgs_occlusion')
    mask_dir = osp.join(output_dir, 'imgs_mask')
    aligned_dir = osp.join(output_dir, 'imgs_aligned')

    begin_time = time.time()
    file_types = ("jpg", "png", "bmp")

    # 创建进程池
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    dector = Facedetection()
    count = 0
    total = len(os.listdir(input_dir))
    for root_dir, dir_list, file_list in os.walk(input_dir):
        file_list2 = [osp.join(root_dir, f) for f in file_list if f[-3:] in file_types]
        if file_list2:
            pool.apply_async(func=do_by_filelist, args=(dector, file_list2, occlusion_dir, mask_dir, aligned_dir))
            # do_by_filelist(dector, file_list2, occlusion_dir, mask_dir, aligned_dir)
            count += 1
            print("{} / {}".format(count, total))

    pool.close()
    pool.join()   # 调用join之前,先调用close函数,执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

    print('finish ！')
    end_time = time.time()
    cost_time = end_time - begin_time
    print('costtime = ', cost_time)


def rename(dir_path):
    import glob
    import os
    import os.path as osp
    file_list = glob.glob("{}/*.png".format(dir_path))

    for i, f in enumerate(file_list):
        bgr = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # if osp.exists(osp.join(osp.dirname(f), "{}.png".format(i+1))):
        #     raise Exception("{} is exist".format(osp.join(osp.dirname(f), "{}.png".format(i+1))))
        # os.rename(f, osp.join(osp.dirname(f), "{}.png".format(i+1)))
        # cv2.imwrite(f, bgr)


if __name__ == "__main__":
    # rename("/home/shengyang/work/hg-git/face_occlusion/resource/arm")
    # exit()
    if len(sys.argv) == 2:
        start_occlusion(input_dir=sys.argv[1])
    if len(sys.argv) == 3:
        start_occlusion(input_dir=sys.argv[1], key=sys.argv[2])
    else:
        start_occlusion(input_dir="resource/test_images")
