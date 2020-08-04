#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author  : 郑祥忠
# @license : (C) Copyright,2013-2019,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : face_classification.py
# @time    : 12/24/19 4:15 PM
# @desc    :
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.12.24 16:15    dylen      v0.1        creation

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np

from torchvision import transforms
import time
import cv2
from torch2trt import TRTModule

import logging
import warnings

from tensorrt import Logger, Runtime
import os


TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


class FaceDataSet(Dataset):
    def __init__(self, imgs: list, transfrom=None):
        self.imgs = imgs
        self.transform = transfrom

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        if self.transform is not None:
            return self.transform(self.imgs[item])
        else:
            return self.imgs[item]


class CvResize(object):
    def __init__(self, dst_size, interpolation=cv2.INTER_CUBIC):
        """
        dst_size=(width, height)
        :param dst_size:
        :param interpolation:
        """
        self.dst_size = dst_size
        self.interpolation = interpolation

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        return cv2.resize(img, dsize=self.dst_size, interpolation=self.interpolation)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += ')'
        return format_string


class DataPrefetcher(object):
    def __init__(self, loader, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.Tensor(np.array(mean)*255).cuda().view(1, 3, 1, 1)
        self.std = torch.Tensor(np.array(std)*255).cuda().view(1, 3, 1, 1)
        self.__preload()

    def __preload(self):
        self.next_input = next(self.loader, None)
        if self.next_input is None:
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        if inputs is not None:
            inputs.record_stream(torch.cuda.current_stream())
        self.__preload()

        if inputs is not None:
            return inputs

        return inputs


def fast_collate(batch):
    imgs = [img for img in batch]
    w = imgs[0].shape[0]
    h = imgs[0].shape[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor


class FaceClass(object):
    def __init__(self, model_path='model/faceclassification.trt'):
        trt_log = Logger(Logger.WARNING)
        if not os.path.exists(model_path):
            raise Exception(f"{model_path} is not exist, please convert from onnx file")

        with open(model_path, 'rb') as fin, Runtime(trt_log) as runtime:
            engine = runtime.deserialize_cuda_engine(fin.read())

        self.model = TRTModule(engine=engine).cuda()
        self.model.input_names = ["input0"]
        self.model.output_names = ["output0"]
        self.softmax = torch.nn.Softmax(dim=1)

        self.inputSize = 112

    @torch.no_grad()
    def classify(self, imgs: list, batchsize=32):
        """
        :param imgs: rgb_image
        :param batchsize:
        :return: 0->frontal_face     1->profile_face
        """
        assert isinstance(imgs, list)
        st = time.time()
        transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((112, 112), interpolation=2),    # resized outside
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
        ])

        face_data = FaceDataSet(imgs, transform)
        loader = DataLoader(dataset=face_data, batch_size=batchsize)
        labels = []

        for inputs in loader:
            inputs = inputs.cuda()
            outputs = self.model(inputs)
            outputs = self.softmax(outputs)
            prob, label = outputs.max(1)
            labels.extend(list(label.data.cpu().numpy()))
        logging.debug(f"classify total:{time.time() - st}")
        return labels

    @torch.no_grad()
    def get_score(self, imgs: list, batchsize=32):
        """
        :param imgs: rgb_image
        :param batchsize:
        :return: numpy arrays
        """
        st = time.time()
        assert isinstance(imgs, list)

        transform = [CvResize(dst_size=(64, 96), interpolation=cv2.INTER_CUBIC)]

        face_data = FaceDataSet(imgs, transforms.Compose(transform))
        dataloader = DataLoader(dataset=face_data, batch_size=batchsize,
                                num_workers=0, collate_fn=fast_collate)

        prefetcher = DataPrefetcher(loader=dataloader, mean=TRAIN_MEAN, std=TRAIN_STD)
        labels = []

        inputs = prefetcher.next()
        while inputs is not None:
            outputs = self.model(inputs)
            outputs = self.softmax(outputs).data.cpu()
            inputs = prefetcher.next()

            prob, label = outputs.max(1)
            labels.extend(list(label.numpy()))
        logging.debug(f"classification total:{time.time() - st}")
        return labels

    @torch.no_grad()
    def classifyNormal(self, imgs: list, batchsize=32):
        """
        :param imgs: rgb_image
        :param batchsize:
        :return: 0->frontal_face     1->profile_face
        """
        assert isinstance(imgs, list)
        st = time.time()
        transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((112, 112), interpolation=2),    # resized outside
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
        ])

        face_data = FaceDataSet(imgs, transform)
        loader = DataLoader(dataset=face_data, batch_size=batchsize)

        label = -1
        for inputs in loader:
            inputs = inputs.cuda()
            outputs = self.model(inputs)
            outputs = self.softmax(outputs)
            prob, label = outputs.max(1)
            label = label.cpu().numpy().item()
        return label

    @torch.no_grad()
    def classify_slow(self, imgs: list, batchsize=32):
        """
        :param imgs: rgb_image
        :param batchsize:
        :return: 0->frontal_face     1->profile_face
        """
        warnings.warn("The 'classify_slow' function is deprecated, use 'classify' instead",
                      DeprecationWarning, 2)

        assert isinstance(imgs, list)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
        ])

        face_data = FaceDataSet(imgs, transform)
        loader = DataLoader(dataset=face_data, batch_size=batchsize)
        labels = []

        for inputs in loader:
            inputs = inputs.cuda()
            outputs = self.model(inputs).data.cpu()
            outputs = self.softmax(outputs)
            prob, label = outputs.max(1)
            labels.extend(list(label.numpy()))
        return labels


if __name__ == "__main__":
    faceClasser = FaceClass(model_path='../model/face_classification_trt.pth')
