#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: crop_and_align.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.04.02 15:17    shengyang      v0.1        creation

#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: demo_trt.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.03.31 19:10    shengyang      v0.1        creation
import numpy as np
import cv2


class Cropper(object):
    def __init__(self):
        self.crop_h, self.crop_w = 112, 112

        # stand land mark points
        reference_lmk = np.matrix([[38.2946, 51.6963],
                                   [73.5318, 51.6963],
                                   [56.0252, 71.7366],
                                   [41.5493, 92.3655],
                                   [70.7299, 92.3655]], dtype=np.float64)

        # SVD para
        self.__mean_lmk = np.mean(reference_lmk, axis=0)
        target_lmk = reference_lmk - self.__mean_lmk
        self.__std_lmk = np.std(target_lmk)
        self.target_lmk = target_lmk / self.__std_lmk
        self.expend = 3  # extend 1//expend each side

    def transformation_from_points(self, input_lmk):
        c1 = np.mean(input_lmk, axis=0)
        input_lmk -= c1
        s1 = np.std(input_lmk)
        input_lmk /= s1

        U, S, Vt = np.linalg.svd(input_lmk.T * self.target_lmk)
        R = (U * Vt).T
        return np.vstack([np.hstack(((self.__std_lmk / s1) * R, self.__mean_lmk.T - (self.__std_lmk / s1) * R * c1.T)),
                          np.matrix([0., 0., 1.])])

    def crop(self, img, boxes, landmarks, align=True, interpolation=cv2.INTER_CUBIC):
        """
        crop face from full image, and resize and align
        :param img: input image, numpy.ndarray
        :param boxes: bounding boxes, N * 4 numpy array, (x1,y1,x2,y2), dtype=np.int
        or list of box
        :param landmarks: N * 5 * 2 numpy array, dtype=np.int
        or list of landmark
        :param align: True or False
        :param interpolation: resize interpolation
        :return: face_list: numpy ndarray, align_face_list: numpy ndarray(if align is True)
        """
        landmarks = landmarks.reshape([-1,5,2])
        new_landmarks = []

        if align is True:
            align_face_list = []
            image_h = img.shape[0]
            image_w = img.shape[1]
            for box, landmark in zip(boxes, landmarks):
                face_w = box[2] - box[0]
                face_h = box[3] - box[1]

                start_h = max(box[1] - face_h // self.expend, 0)
                start_w = max(box[0] - face_w // self.expend, 0)
                end_h = min(box[3] + face_h // self.expend, image_h)
                end_w = min(box[2] + face_w // self.expend, image_w)

                face_img_bigger = img[start_h:end_h, start_w:end_w, :]
                oripoint = np.array([start_w, start_h]).reshape((1, 2))
                landmark = landmark - oripoint

                input_lmk = np.matrix(landmark, dtype=np.float64)
                transfrom_matrix = self.transformation_from_points(input_lmk)
                aligned = cv2.warpAffine(src=face_img_bigger, M=transfrom_matrix[:2],
                                         dsize=(self.crop_w, self.crop_h), flags=interpolation)
                align_face_list.append(aligned)
                new_landmarks.append(landmark)

            face_list = [cv2.resize(img[box[1]:box[3], box[0]:box[2], :], (self.crop_h, self.crop_w), interpolation)
                         for box in boxes]
            return face_list, align_face_list, new_landmarks
        else:
            face_list = [cv2.resize(img[box[1]:box[3], box[0]:box[2], :], (self.crop_h, self.crop_w), interpolation)
                         for box in boxes]
            return face_list


if __name__ == "__main__":
    from glob import iglob
    import threading

    cropper = Cropper()
    for one_file in iglob("/home/shengyang/work/git/face_detection/CenterFace/prj-python/o*.jpg"):
        image = cv2.imread(one_file)
        boxes, scores, quality_list, angle, landmarks = facedetect.detect(image)

        # just crop and resize not align
        # faces = cropper.crop(image_bgr, boxes, landmarks, align=False, interpolation=cv2.INTER_LINEAR)
        # print(1000 * (time.time()-st))
        # for face in faces:
        #     cv2.imshow('face', face)
        #     cv2.waitKey(0)

        # crop, align and resize
        faces, aligned_faces = cropper.crop(image, boxes, landmarks, align=True, interpolation=cv2.INTER_LINEAR)
