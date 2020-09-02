#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: demo_trt.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.03.31 19:10    shengyang      v0.1        creation
# 2020.04.01 09:10    lingangxin     v0.2        torch2trt

import os
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

from tensorrt import Logger, Runtime
from torch2trt.torch2trt import TRTModule
import cv2
import torch
import numpy as np


class Facedetection(object):
    def __init__(self, trt_file="model/facedetection.trt"):
        trt_log = Logger(Logger.WARNING)
        if not os.path.exists(trt_file):
            raise Exception(f"{trt_file} is not exist, please convert from onnx file")

        with open(trt_file, 'rb') as fin, Runtime(trt_log) as runtime:
            engine = runtime.deserialize_cuda_engine(fin.read())

        self.model = TRTModule(engine=engine).cuda()
        self.model.input_names = ["input.1"]
        self.model.output_names = ["537", "538", "539", "540"]

        self.input_shapes = []
        self.output_shapes = []
        for binding in engine:
            if engine.binding_is_input(binding):
                self.input_shapes.append(tuple([engine.max_batch_size] + list(engine.get_binding_shape(binding))))
            else:
                self.output_shapes.append(tuple([engine.max_batch_size] + list(engine.get_binding_shape(binding))))

        if len(self.input_shapes) != 1:
            logging.warning('Only one input data is supported.')

        self.input_shape = self.input_shapes[0]
        self.input_channel = self.input_shape[1]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        self.threshold = 0.4    # 检测分数阈值,越大错检越少,漏检越多
        self.nms_thresh = 0.5   # nms阈值,越小重复框越少

        self.executer = ThreadPoolExecutor(max_workers=4)
        self.thread_lock = threading.Lock()

    def postprocess(self, heatmap, landmark, offset, scale, resize_scale):
        """

        :param heatmap:   N  *  1 * 120 * 200
        :param landmark:  N  * 10 * 120 * 200
        :param offset:    N  *  2 * 120 * 200
        :param scale:     N  *  2 * 120 * 200
        :param resize_scale:
        :return:
        """
        all_boxes = []
        all_lms = []
        all_scores = []
        for one_heatmap, one_landmark, one_offset, one_scale, one_resize_scale in zip(heatmap, landmark, offset, scale, resize_scale):
            one_heatmap = np.squeeze(one_heatmap)
            scale0, scale1 = one_scale[0, :, :], one_scale[1, :, :]
            offset0, offset1 = one_offset[0, :, :], one_offset[1, :, :]
            x_indexs, y_indexs = np.where(one_heatmap > self.threshold)
            boxes, lms = [], []
            scores = []

            for x_index, y_index in zip(x_indexs, y_indexs):
                s0, s1 = np.exp(scale0[x_index, y_index]) * 4, np.exp(scale1[x_index, y_index]) * 4
                o0, o1 = offset0[x_index, y_index], offset1[x_index, y_index]
                s = one_heatmap[x_index, y_index]
                x1, y1 = max(0, (y_index + o1 + 0.5) * 4 - s1 / 2), max(0, (x_index + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, self.input_width), min(y1, self.input_height)
                boxes.append(
                    [int(x1), int(y1), int(min(x1 + s1, self.input_width)), int(min(y1 + s0, self.input_height))])
                scores.append(float(s))
                lm = []
                for j in range(5):
                    lm.append(one_landmark[j * 2 + 1, x_index, y_index] * s1 + x1)
                    lm.append(one_landmark[j * 2, x_index, y_index] * s0 + y1)
                lms.append(lm)

            if boxes:
                keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.threshold, nms_threshold=self.nms_thresh)
                keep = keep.flatten().tolist()
                boxes = np.asarray(boxes, dtype=np.float32)[keep, :]
                scores = np.asarray(scores, dtype=np.float32)[keep].reshape((-1))
                lms = np.asarray(lms, dtype=np.float32)
                lms = lms[keep, :]

                boxes = boxes / one_resize_scale
                lms = lms / one_resize_scale
                all_boxes.append(boxes.astype(np.int))
                all_scores.append(scores)
                all_lms.append(lms.astype(np.int).reshape((-1, 5, 2)))
            else:
                all_boxes.append([])
                all_scores.append([])
                all_lms.append([])

        return all_boxes, all_scores, all_lms

    def fix_box(self, boxes, max_w, max_h):
        """
        adjust target rectangular'box into square'box
        """
        for one_box in boxes:
            diff = (one_box[3] - one_box[1]) - (one_box[2] - one_box[0])
            if diff > 0:
                one_box[0] -= diff // 2
                one_box[2] += diff - (diff // 2)
                one_box[0] = max(one_box[0], 0)
                one_box[2] = min(one_box[2], max_w)
            elif diff < 0:
                one_box[1] += diff // 2
                one_box[3] -= diff - (diff // 2)
                one_box[1] = max(one_box[1], 0)
                one_box[3] = min(one_box[3], max_h)

        return boxes

    def imagelist2batch(self, img_list):
        n = len(img_list)
        input_batch = np.zeros((n, self.input_height, self.input_width, self.input_channel), dtype=np.uint8)
        resize_scale = []
        for index, image_rgb in enumerate(img_list):
            image_h, image_w, image_c = image_rgb.shape
            if image_h / image_w > self.input_height / self.input_width:
                one_resize_scale = self.input_height / image_h
                input_image = cv2.resize(image_rgb, (0, 0), fx=one_resize_scale, fy=one_resize_scale)
                input_batch[index, :, 0:input_image.shape[1], :] = input_image
            else:
                one_resize_scale = self.input_width / image_w
                input_image = cv2.resize(image_rgb, (0, 0), fx=one_resize_scale, fy=one_resize_scale)
                input_batch[index, 0:input_image.shape[0], :, :] = input_image
            resize_scale.append(one_resize_scale)

        input_batch = input_batch.transpose([0, 3, 1, 2])
        input_batch = np.array(input_batch, dtype=np.float32, order='C')
        input_batch_tensor = torch.Tensor(input_batch)

        return input_batch_tensor, resize_scale

    def face_detect(self, image_rgb_list, debug=False):
        """
        input image_rgb list detect face from each image
        :param image_rgb: input image_list
        :param debug: if debug is True save the result image
        :return:
            coordinations: boxes, [[[x1,y1,x2,y2], [x1,y1,x2,y2]],   # image1 result face boxes
                                   [[x1,y1,x2,y2],[x1,y1,x2,y2], [x1,y1,x2,y2]],  # image2 result face boxes
                                   ... # imageN result face boxes, if not found any face in this image, it will be []
                                  ]
            scores, [s1, s2, s3, ...]
            all_landmarks: landmarks, [left_eye_x, left_eye_y,     right_eye_x, right_eye_y,
                                                           nose_x, nose_y,
                                       mouth_left_x, mouth_left_y, mouth_right_x, mouth_right_y]
        """
        st = time.time()
        input_batch_tensor, resize_scale = self.imagelist2batch(image_rgb_list)
        logging.debug(f"pre process: {1000 * (time.time() - st):.3f}")

        st = time.time()
        heatmap, scale, offset, landmarks = self.model(input_batch_tensor.cuda())
        logging.debug(f"process: {1000 * (time.time() - st):.3f}")

        heatmap = heatmap.cpu().numpy()
        scale = scale.cpu().numpy()
        offset = offset.cpu().numpy()
        landmarks = landmarks.cpu().numpy()

        st = time.time()
        all_boxes, all_scores, all_landmarks = self.postprocess(heatmap, landmarks, offset, scale, resize_scale=resize_scale)
        logging.debug(f"post process: {1000 * (time.time() - st):.3f}")
        all_face_rgb = []

        for boxes, scores, landmarks, image_rgb in zip(all_boxes, all_scores, all_landmarks, image_rgb_list):
            image_h, image_w, _ = image_rgb.shape
            boxes = self.fix_box(boxes, image_w, image_h)

            rgb_list = [cv2.resize(image_rgb[box[1]: box[3], box[0]: box[2], :], (112, 112)) for box in boxes]
            all_face_rgb.extend(rgb_list)

        if debug is True:
            colors_hp = [(0, 0, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 0)]
            for image_rgb, boxes, scores, landmarks in zip(image_rgb_list, all_boxes, all_scores, all_landmarks):
                img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                image_h, image_w, _ = image_rgb.shape

                for box, land_mark in zip(boxes, landmarks):
                    cv2.rectangle(img_bgr, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=3)
                    for color_index, point in enumerate(land_mark):
                        cv2.circle(img_bgr, (point[0], point[1]), 3, colors_hp[color_index], -1)

                cv2.imwrite(f"./saved/{time.time()}.jpg", img_bgr)

        return all_boxes, all_scores, all_landmarks


if __name__ == "__main__":
    pass
