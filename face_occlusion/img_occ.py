import cv2
import numpy as np
import random
from scipy import optimize
import math
import matplotlib.pyplot as plt
# from PIL import Image
# import os
import copy

import glob


# background_dir = "resource/backgroud"
# background_file_list = glob.glob("{}/*.jpg".format(background_dir))
# background_list = [cv2.imread(one_file) for one_file in background_file_list]

mask_dir = "resource/mask"
mask_file_list = glob.glob("{}/7*.png".format(mask_dir))
mask_list = [cv2.imread(one_file, cv2.IMREAD_UNCHANGED) for one_file in mask_file_list]

# arm_dir = "resource/arm"
# arm_file_list = glob.glob("{}/*.png".format(arm_dir))
# arm_list = [cv2.imread(one_file, cv2.IMREAD_UNCHANGED) for one_file in arm_file_list]

# hand_dir = "resource/hand"
# hand_file_list = glob.glob("{}/*.png".format(hand_dir))
# hand_list = [cv2.imread(one_file, cv2.IMREAD_UNCHANGED) for one_file in hand_file_list]

# scarf_dir = "resource/scarf"
# scarf_file_list = glob.glob("{}/*.png".format(scarf_dir))
# scarf_list = [cv2.imread(one_file, cv2.IMREAD_UNCHANGED) for one_file in scarf_file_list]

# hat_dir = "resource/hat"
# hat_file_list = glob.glob("{}/*.png".format(hat_dir))
# hat_list = [cv2.imread(one_file, cv2.IMREAD_UNCHANGED) for one_file in hat_file_list]

# classes_dir = "resource/classes"
# classes_file_list = glob.glob("{}/*.png".format(classes_dir))
# classes_list = [cv2.imread(one_file, cv2.IMREAD_UNCHANGED) for one_file in classes_file_list]


def get_angle_from_keypoint(keypoints) -> int:
    """
    get_angle_from_keypoint
    :param keypoints: l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y, mouth_l_x, mouth_l_y, mouth_r_x, mouth_r_y
    :return: angle int
    """
    # 直线拟合与绘制
    center_of_eyes_x = (keypoints[0] + keypoints[2]) / 2
    center_of_eyes_y = (keypoints[1] + keypoints[3]) / 2
    nose_x = keypoints[4]
    nose_y = keypoints[5]
    center_of_mouth_x = (keypoints[6] + keypoints[8]) / 2
    center_of_mouth_y = (keypoints[7] + keypoints[9]) / 2
    xs = [center_of_eyes_x, nose_x, center_of_mouth_x]
    ys = [center_of_eyes_y, nose_y, center_of_mouth_y]

    a, b = optimize.curve_fit(lambda x, a, b: a*x + b, xs, ys)[0]

    if a > 0:
        angle = 90 - math.atan(a) / math.pi * 180
    else:
        angle = -90 - math.atan(a) / math.pi * 180

    return int(angle)


def occ_1(img, box):
    # 1/4遮挡
    rows, cols, cns = img.shape
    sticker = copy.copy(random.choice(background_list))
    resize_rate = random.uniform(0.5, 2)
    sticker = cv2.resize(src=sticker, dsize=(0, 0), fx=resize_rate, fy=resize_rate)
    mod = random.randint(0, 3)
    img_binary = np.empty(shape=(rows, cols), dtype=np.uint8)
    img_binary.fill(255)

    if mod == 0:
        sub_part = img[0:box[4], 0:box[5], :]
        mask = img_binary[0:box[4], 0:box[5]]
        random_area_fill(mask)
        boxes_occ = np.asarray([1, 0, 0, box[4], box[5], 0, 0])
    elif mod == 1:
        sub_part = img[0:box[4], box[5]:cols, :]  # 右上角
        mask = img_binary[0:box[4], box[5]:cols]
        random_area_fill(mask)
        boxes_occ = np.asarray([1, 0, box[5], box[4], cols, 0, 0])
    elif mod == 2:
        sub_part = img[box[4]:rows, 0:box[5], :]  # 左下角
        mask = img_binary[box[4]:rows, 0:box[5]]
        random_area_fill(mask)
        boxes_occ = np.asarray([1, box[4], 0, rows, box[5], 0, 0])
    else:
        sub_part = img[box[4]:rows, box[5]:cols, :]  # 右下角
        mask = img_binary[box[4]:rows, box[5]:cols]
        random_area_fill(mask)
        boxes_occ = np.asarray([1, box[4], box[5], rows, cols, 0, 0])

    mask_w = mask.shape[0]
    mask_h = mask.shape[1]
    src_w = sticker.shape[0]
    src_h = sticker.shape[1]
    start_w = random.randint(0, src_w - mask_w)
    end_w = start_w + mask_w
    start_h = random.randint(0, src_h - mask_h)
    end_h = start_h + mask_h

    cv2.copyTo(src=sticker[start_w: end_w, start_h: end_h, :], dst=sub_part, mask=255-mask)

    return img_binary, img, boxes_occ


def occ_2(img, box):
    ### 左右1/2遮挡 ###
    rows, cols, cns = img.shape
    mod = random.randint(0, 1)
    img_binary = np.empty(shape=(rows, cols), dtype=np.uint8)
    img_binary.fill(255)

    if mod == 0:
        img[:, 0:box[5], :] = 0
        img_binary[:, 0:box[5], :] = 0
        boxes_occ = np.asarray([2, 0, 0, rows, box[5], 0, 0])
    else:
        img[:, box[5]:cols, :] = 0
        img_binary[:, box[5]:cols, :] = 0
        boxes_occ = np.asarray([2, 0, box[5], rows, cols, 0, 0])
    return img_binary, img, boxes_occ


def occ_3(input_image_rgb, keypoint, path="resource/mask"):
    # 口罩贴纸,手掌贴纸,围巾贴纸
    # img_list = os.listdir(path)
    # random_num = random.randint(0, len(img_list) - 1)
    # img_dir = os.path.join(path, img_list[random_num])
    # sticker_image = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
    # # sticker_image = cv2.imread("resource/mask/11.png", cv2.IMREAD_UNCHANGED)
    # flip_num = random.randint(0, 1)
    # if flip_num == 1:
    #     sticker_image = cv2.flip(sticker_image, 1)

    sticker_image = copy.copy(random.choice(mask_list))

    height, width, cns = input_image_rgb.shape

    angle = get_angle_from_keypoint(keypoint)

    sticker_height, sticker_witdh, _ = sticker_image.shape
    face_width = max(int((keypoint[2] - keypoint[0]) * 2.3), 80)   # face width = 2.3 * eye witdh
    scale = [0.8 * face_width / sticker_height, 1.2 * face_width / sticker_height]
    sticker_image = random_transform(sticker_image, angle=(angle-5, angle+5), scale=scale)

    # 获取贴图的透明层
    alpha_channel = sticker_image[:, :, -1]
    sticker_rgb = cv2.cvtColor(sticker_image, cv2.COLOR_BGRA2RGB)

    # 贴图遮挡的mask,初始化为255
    image_binary = np.empty(shape=(height, width), dtype=np.uint8)
    image_binary.fill(255)

    # 贴图resize到与人脸相似的尺寸
    ori_mask_h, ori_mask_w = alpha_channel.shape
    sticker_w = random.randint(int(face_width*0.9), face_width)
    sticker_h = int(sticker_w / ori_mask_w * ori_mask_h)

    alpha_channel = cv2.resize(alpha_channel, (sticker_w, sticker_h))
    sticker_rgb = cv2.resize(sticker_rgb, (sticker_w, sticker_h))

    left_eye_y = keypoint[1]
    right_eye_y = keypoint[3]
    nose_y = keypoint[5]

    # 计算贴图拷贝到源图时的起始位置和终止位置
    # 左右居中,贴图的上边对齐鼻子与眼睛的中间高度
    start_copy_x = (width - sticker_w) // 2
    start_copy_y = (nose_y + (left_eye_y + right_eye_y) // 2) // 2
    end_copy_x = sticker_w + start_copy_x
    end_copy_y = min((start_copy_y + sticker_h), height)

    # 截取直接拷贝的部分
    src_sticker_rgb = sticker_rgb[0: min((height-start_copy_y), sticker_h), :, :]
    src_sticker_mask = alpha_channel[0: min((height-start_copy_y), sticker_h), :]
    dst_image_rgb = input_image_rgb[start_copy_y: end_copy_y, start_copy_x: end_copy_x, :]

    # 根据透明层进行拷贝,透明的部分保留原来的颜色
    cv2.copyTo(src_sticker_rgb, src_sticker_mask, dst_image_rgb)

    src_img_zero = np.zeros_like(src_sticker_mask, dtype=np.uint8)
    dst_img = image_binary[start_copy_y: end_copy_y, start_copy_x: end_copy_x]
    # 根据透明层进行拷贝生成mask,非透明的部分被设置为黑色
    cv2.copyTo(src_img_zero, src_sticker_mask, dst_img)

    return image_binary, input_image_rgb, None


def occ_4(img, box):
    # 遮挡眼睛鼻子之外部分
    rows, cols, cns = img.shape
    scale_l = int(box[1] / 2)
    scale_r = int((cols - box[3]) / 2)
    scale_u = int(rows / 10)
    scale_d = int(rows / 10)
    x1 = box[0] - scale_u
    y1 = box[1] - scale_l
    x2 = box[2] - scale_u
    y2 = box[3] + scale_r
    x3 = box[4] + scale_d
    y3 = box[5]
    box1 = np.asarray([[[y1, x1], [y2, x2], [y3, x3]]])
    img_copy = img.copy()
    img_binary = np.empty(shape=(rows, cols), dtype=np.uint8)
    img_binary.fill(255)

    img_binary_copy = img_binary.copy()
    cv2.fillPoly(img_copy, box1, 0)
    cv2.fillPoly(img_binary_copy, box1, 0)
    new_img = img - img_copy
    new_img_binary = img_binary - img_binary_copy
    boxes_occ = np.asarray([4, x1, y1, x2, y2, x3, y3])
    return new_img_binary, new_img, boxes_occ


def occ_5(img, box):
    # 遮挡眼睛鼻子
    rows, cols, cns = img.shape
    img_binary = np.empty(shape=(rows, cols), dtype=np.uint8)
    img_binary.fill(255)

    scale_l = int(box[1] / 2)
    scale_r = int((cols - box[3]) / 2)
    scale_u = int(rows / 10)
    scale_d = int(rows / 10)
    x1 = box[0] - scale_u
    y1 = box[1] - scale_l
    x2 = box[2] - scale_u
    y2 = box[3] + scale_r
    x3 = box[4] + scale_d
    y3 = box[5]
    box1 = np.asarray([[[y1, x1], [y2, x2], [y3, x3]]])
    cv2.fillPoly(img, box1, 0)
    cv2.fillPoly(img_binary, box1, 0)
    boxes_occ = np.asarray([5, x1, y1, x2, y2, x3, y3])
    return img_binary, img, boxes_occ


def occ_6(img, box):
    # 遮挡除眼睛之外的部分
    rows, cols, cns = img.shape
    img_binary = np.empty(shape=(rows, cols), dtype=np.uint8)
    img_binary.fill(255)

    scale_l = int(box[1] / 2)
    scale_r = int((cols - box[3]) / 2)
    scale_u = int(rows / 10)
    x1 = box[0] - scale_u
    y1 = box[1] - scale_l
    x2 = box[2] + scale_u
    y2 = box[3] + scale_r
    img_eyes = img[x1:x2, y1:y2, :]
    img_binary_eyes = img_binary[x1:x2, y1:y2, :]
    img_new = np.zeros((rows, cols), dtype=np.uint8)
    img_new = cv2.cvtColor(img_new, cv2.COLOR_GRAY2BGR)
    img_binary_new = np.zeros((rows, cols), dtype=np.uint8)
    img_binary_new = cv2.cvtColor(img_binary_new, cv2.COLOR_GRAY2BGR)
    img_new[x1:x2, y1:y2, :] = img_eyes
    img_binary_new[x1:x2, y1:y2, :] = img_binary_eyes
    boxes_occ = np.asarray([6, x1, y1, x2, y2, 0, 0])
    return img_binary_new, img_new, boxes_occ


def occ_7(img, box, path="resource/classes"):
    # 眼镜贴纸
    # img_list = os.listdir(path)
    # random_num = random.randint(0, len(img_list) - 1)
    # img_dir = os.path.join(path, img_list[random_num])
    # classes = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
    # # classes = cv2.imread("resource/classes/11.png", cv2.IMREAD_UNCHANGED)
    # flip_num = random.randint(0, 1)
    # if flip_num == 1:
    #     classes = cv2.flip(classes, 1)

    classes = copy.copy(random.choice(classes_list))
    if random.choice([True, False]):
        classes = cv2.flip(classes, 1)

    classes = random_transform(classes, angle=(-5, 5), scale=(0.9, 1.1), flip=True, blur=True,
                               noise=True, hsv_transform=False, contrast=False)

    c1, c2, c3, mask = cv2.split(classes)

    rows, cols, cns = img.shape
    img_binary = np.empty(shape=(rows, cols), dtype=np.uint8)
    img_binary.fill(255)

    scale_l = int(box[1] / 2)
    scale_r = int((cols - box[3]) / 2)
    scale_u = int(rows / 10)
    x1 = box[0] - scale_u
    y1 = box[1] - scale_l
    x2 = box[2] + scale_u
    y2 = box[3] + scale_r

    # mask = cv2.resize(mask, (cols, x2-x1))
    # print(mask.shape)
    # classes = cv2.resize(classes, (y2 - y1, x2 - x1))
    # classes = cv2.cvtColor(classes, cv2.COLOR_BGRA2BGR)
    # cv2.copyTo(classes, mask[:], img[x1:x2, y1:y2, :])
    if box[2] >= box[0]:
        mask = cv2.resize(mask, (cols, x2 - x1))
        classes = cv2.resize(classes, (cols, x2 - x1))
        classes = cv2.cvtColor(classes, cv2.COLOR_BGRA2BGR)
        if classes.shape[0] % 2:
            mask = cv2.resize(mask, (cols, classes.shape[0] - 1))
            classes = cv2.resize(classes, (cols, classes.shape[0] - 1))
        half_h = int(classes.shape[0] // 2)

        if half_h > box[0]:
            mask = cv2.resize(mask, (cols, 2 * box[0]))
            classes = cv2.resize(classes, (cols, 2 * box[0]))
            half_h = int(classes.shape[0] // 2)
        strat_row = box[2] - half_h
        end_row = box[2] + half_h

        cv2.copyTo(classes, mask[:], img[strat_row+scale_u:end_row+scale_u, :, :])
        img_zero = np.zeros((rows, cols), dtype=np.uint8)
        cv2.bitwise_and(img_binary[strat_row+scale_u:end_row+scale_u, :], img_zero[strat_row+scale_u:end_row+scale_u, :],
                        dst=img_binary[strat_row+scale_u:end_row+scale_u, :], mask=mask)
    elif box[2] < box[0]:
        temp_height = abs(box[2] - box[0]) + 2 * scale_u
        mask = cv2.resize(mask, (cols, temp_height))
        classes = cv2.resize(classes, (cols, temp_height))
        classes = cv2.cvtColor(classes, cv2.COLOR_BGRA2BGR)
        if classes.shape[0] % 2:
            mask = cv2.resize(mask, (cols, classes.shape[0] - 1))
            classes = cv2.resize(classes, (cols, classes.shape[0] - 1))
        half_h = int(classes.shape[0] // 2)
        if half_h > box[2]:
            mask = cv2.resize(mask, (cols, 2 * box[2]))
            classes = cv2.resize(classes, (cols, 2 * box[2]))
            half_h = int(classes.shape[0] // 2)
        strat_row = box[0] - half_h
        end_row = box[0] + half_h
        cv2.copyTo(classes, mask[:], img[strat_row+scale_u:end_row+scale_u, :, :])
        img_zero = np.zeros((rows, cols), dtype=np.uint8)
        cv2.bitwise_and(img_binary[strat_row+scale_u:end_row+scale_u, :], img_zero[strat_row+scale_u:end_row+scale_u, :],
                        dst=img_binary[strat_row+scale_u:end_row+scale_u, :], mask=mask)
    boxes_occ = np.asarray([7, x1, y1, x2, y2, 0, 0])
    return img_binary, img, boxes_occ


# def occ_8(img_binary, img, box):
#     ### 遮挡额头 ###
#     rows, cols, cns = img.shape
#     scale_u = int(rows / 20)
#     x1 = box[0] - scale_u
#     x2 = box[2] - scale_u
#     x = int((x1 + x2) / 2)
#     img[0:x, :, :] = 0
#     img_binary[0:x, :, :] = 0
#     boxes_occ = np.asarray([8, 0, 0, x, cols, 0, 0])
#     return img_binary, img, boxes_occ


def occ_8(input_image, box, path="resource/hat"):
    # 帽子贴纸, 手臂贴纸
    sticker_type = random.choice(['hat', 'arm'])
    if sticker_type is 'hat':
        sticker_image = copy.copy(random.choice(hat_list))
    else:
        sticker_image = copy.copy(random.choice(arm_list))
    height, width, cns = input_image.shape

    sticker_image = random_transform(sticker_image, scale=(0.85, 1.0))

    mask = sticker_image[:, :, -1]

    sticker_bgr = cv2.cvtColor(sticker_image, cv2.COLOR_BGRA2BGR)
    image_binary = np.empty(shape=(height, width), dtype=np.uint8)
    image_binary.fill(255)

    scale_l = int(box[1] / 2)
    scale_r = int((width - box[3]) / 2)
    scale_u = int(height / 10)

    ori_mask_h, ori_mask_w = mask.shape

    if sticker_type is 'arm':
        mask_h = random.randint(int(height*0.33), int(height*0.4))
        mask_w = int(mask_h / ori_mask_h * ori_mask_w)
    elif sticker_type is 'hat':
        mask_w = random.randint(int(width * 0.9), width)
        mask_h = int(mask_w / ori_mask_w * ori_mask_h)
    else:
        raise Exception("not support sticker_type: {}".format(sticker_type))

    mask = cv2.resize(mask, (mask_w, mask_h))
    sticker_bgr = cv2.resize(sticker_bgr, (mask_w, mask_h))

    left_eye_x = box[1] - scale_l
    left_eye_y = box[0] - scale_u
    right_eye_x = box[3] + scale_r
    right_eye_y = box[2] + scale_u

    start_copy_x = 0
    start_copy_y = 0
    end_copy_x = min(mask_w, width)
    end_copy_y = min(mask_h, height)

    src_sticker_bgr = sticker_bgr[0: min(height, mask_h), 0: min(width, mask_w), :]
    src_sticker_mask = mask[0: min(height, mask_h), 0: min(width, mask_w)]
    dst_image = input_image[start_copy_y: end_copy_y, start_copy_x: end_copy_x, :]

    cv2.copyTo(src_sticker_bgr, src_sticker_mask, dst_image)
    cv2.imwrite("out.jpg", input_image)

    src_img_zero = np.zeros_like(src_sticker_mask, dtype=np.uint8)

    dst_img = image_binary[start_copy_y: end_copy_y, start_copy_x: end_copy_x]

    cv2.copyTo(src_img_zero, src_sticker_mask, dst_img)
    cv2.imwrite("out_mask.jpg", image_binary)

    boxes_occ = np.asarray([7, left_eye_x, left_eye_y, right_eye_x, right_eye_y, 0, 0])

    return image_binary, input_image, boxes_occ


# TODO FIX area rate
def occ_random_area(input_image, box, rate=0.3, path="resource/hat"):
    rows, cols, cns = input_image.shape
    sticker = copy.copy(random.choice(background_list))
    resize_rate = random.uniform(0.5, 2)
    sticker = cv2.resize(src=sticker, dsize=(0, 0), fx=resize_rate, fy=resize_rate)
    mod = random.randint(0, 3)
    img_binary = np.empty(shape=(rows, cols), dtype=np.uint8)
    img_binary.fill(255)

    if mod == 0:
        sub_part = input_image[0:box[4], 0:box[5], :]
        mask = img_binary[0:box[4], 0:box[5]]
        random_area_fill(mask)
        boxes_occ = np.asarray([1, 0, 0, box[4], box[5], 0, 0])
    elif mod == 1:
        sub_part = input_image[0:box[4], box[5]:cols, :]  # 右上角
        mask = img_binary[0:box[4], box[5]:cols]
        random_area_fill(mask)
        boxes_occ = np.asarray([1, 0, box[5], box[4], cols, 0, 0])
    elif mod == 2:
        sub_part = input_image[box[4]:rows, 0:box[5], :]  # 左下角
        mask = img_binary[box[4]:rows, 0:box[5]]
        random_area_fill(mask)
        boxes_occ = np.asarray([1, box[4], 0, rows, box[5], 0, 0])
    else:
        sub_part = input_image[box[4]:rows, box[5]:cols, :]  # 右下角
        mask = img_binary[box[4]:rows, box[5]:cols]
        random_area_fill(mask)
        boxes_occ = np.asarray([1, box[4], box[5], rows, cols, 0, 0])

    mask_w = mask.shape[0]
    mask_h = mask.shape[1]
    src_w = sticker.shape[0]
    src_h = sticker.shape[1]
    start_w = random.randint(0, src_w - mask_w)
    end_w = start_w + mask_w
    start_h = random.randint(0, src_h - mask_h)
    end_h = start_h + mask_h

    cv2.copyTo(src=sticker[start_w: end_w, start_h: end_h, :], dst=sub_part, mask=255-mask)

    return img_binary, input_image, boxes_occ


def face_occlusion(input_img_rgb, keypoint, mod=None):
    img_rgb = copy.copy(input_img_rgb)
    rows, cols, cns = img_rgb.shape
    boxes_occ = []

    if mod is None:
        # mod = random.choice([1, 7])
        mod = 3
    # mod = 1
    # occlusion_mod == 1, 以鼻尖为起点,图片四角(随机一个角)为终点的矩形遮挡
    # occlusion_mod == 2, 以鼻尖竖线为起始边,图片左右两边(随机一边)为终边的矩形遮挡
    # occlusion_mod == 3, 口罩贴图, 围巾贴图, 手掌的遮挡
    # occlusion_mod == 4, 遮挡眼睛鼻子之外部分
    # occlusion_mod == 5, 遮挡双眼和鼻子的三角区域
    # occlusion_mod == 6, 遮挡除眼睛之外的部分
    # occlusion_mod == 7, 墨镜贴图遮挡
    # occlusion_mod == 8, 帽子贴图, 手臂贴图, 手掌遮挡
    # else, 不遮挡

    if mod == 1:
        img_mask, img_occ, boxes_occ = occ_1(img_rgb, keypoint)
    elif mod == 2:
        img_mask, img_occ, boxes_occ = occ_2(img_rgb, keypoint)
    elif mod == 3:
        img_mask, img_occ, boxes_occ = occ_3(img_rgb, keypoint)
    elif mod == 4:
        img_mask, img_occ, boxes_occ = occ_4(img_rgb, keypoint)
    elif mod == 5:
        img_mask, img_occ, boxes_occ = occ_5(img_rgb, keypoint)
    elif mod == 6:
        img_mask, img_occ, boxes_occ = occ_6(img_rgb, keypoint)
    elif mod == 7:
        img_mask, img_occ, boxes_occ = occ_7(img_rgb, keypoint)
    elif mod == 8:
        img_mask, img_occ, boxes_occ = occ_8(img_rgb, keypoint)
    else:
        img_occ = img_rgb
        img_mask = np.empty(shape=(rows, cols), dtype=np.uint8)
        img_mask.fill(255)

    return img_mask, img_occ, boxes_occ, mod


def random_transform(sticker_image, angle=(-10, 10), scale=(0.8, 1.0), flip=True, blur=True,
                     noise=True, hsv_transform=True, contrast=True):
    h = sticker_image.shape[0]
    w = sticker_image.shape[1]
    center = (w//2, h//2)
    angle_ = min(max(random.randint(angle[0], angle[1]), 15), -15)
    scale_ = random.uniform(scale[0], scale[1])
    matrix = cv2.getRotationMatrix2D(center=center, angle=angle_, scale=scale_)
    sticker_image = cv2.warpAffine(sticker_image, matrix, dsize=(w, h))

    if flip and random.random() > 0.5:
        sticker_image = sticker_image[:, ::-1]

    sticker_image_bgr = cv2.cvtColor(sticker_image, cv2.COLOR_BGRA2BGR)

    if contrast and random.random() > 0.5:
        slope = random.uniform(0.7, 1.1)   # 对比度调整系数
        sticker_image_bgr = sticker_image_bgr * slope
        sticker_image_bgr = np.clip(sticker_image_bgr, 0, 255)

    # 高斯噪声图像
    if noise and random.random() > 0.5:
        gauss = np.random.normal(loc=0, scale=5, size=sticker_image_bgr.shape)
        sticker_image_bgr = np.clip(sticker_image_bgr + gauss, 0, 255)

    # 高斯模糊
    if blur and random.random() > 0.5:
        k = random.choice([3, 5, 7, 9, 11])
        sigma = 2 * random.random()
        sticker_image_bgr = cv2.GaussianBlur(src=sticker_image_bgr, ksize=(k, k), sigmaX=sigma)

    sticker_image_bgr = sticker_image_bgr.astype(np.uint8)

    # HSV变换
    if hsv_transform and random.random() > 0.5:
        hue_delta = np.random.randint(-3, 3)     # 色调变化比例范围
        sat_mult = 1 + np.random.uniform(-0.2, 0.2)  # 饱和度变化比例范围
        val_mult = 1 + np.random.uniform(-0.2, 0.2)  # 明度变化比例范围
        img_hsv = cv2.cvtColor(sticker_image_bgr, cv2.COLOR_BGR2HSV).astype(np.float)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        img_hsv[:, :, 1] *= sat_mult
        img_hsv[:, :, 2] *= val_mult
        img_hsv[img_hsv > 255] = 255
        sticker_image_bgr = cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

    sticker_image[:, :, 0: 3] = sticker_image_bgr

    return sticker_image


def random_area_fill(input_image, color=(0, 0, 0)):
    # Create a black image
    # h = random.randint(30, 70)
    # w = random.randint(30, 70)
    # print(h, w)
    # input_image = np.zeros((h, w), np.uint8)

    w = input_image.shape[0]
    h = input_image.shape[1]
    half_w = w // 2
    half_h = h // 2

    mod = random.choice([1, 2, 3, 4])
    if mod == 1:
        # draw rectangle
        pt1 = (random.randint(0, w//10), h - random.randint(0, h//10))
        pt2 = (w - random.randint(0, 0), random.randint(0, 0))
        cv2.rectangle(input_image, pt1=pt1, pt2=pt2, color=color, thickness=-1)
    elif mod == 2:
        # draw ellipse
        cent = (random.randint(half_w-half_w//2, half_w+half_w//2),
                random.randint(half_h-half_h//2, half_h+half_h//2))
        axes = (random.randint(half_w-half_w//4, half_w+half_w//4),
                random.randint(half_h-half_h//4, half_h+half_h//4))
        angle = random.randint(0, 180)
        cv2.ellipse(input_image, center=cent, axes=axes, angle=angle,
                    startAngle=0, endAngle=360, color=color, thickness=-1)
    else:
        # draw circle
        cent = (random.randint(0, half_w), random.randint(half_h - h // 4, half_h))
        radius = random.randint(min(half_w, half_h), max(half_w, half_h)*2)
        cv2.circle(input_image, center=cent, radius=radius, color=color, thickness=-1)

    return input_image
