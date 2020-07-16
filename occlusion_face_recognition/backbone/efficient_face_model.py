#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: face_model.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.06.03 17:21    shengyang      v0.1        creation

from efficientnet_pytorch import EfficientNet

from torch.nn import Linear, BatchNorm1d, BatchNorm2d
from torch.nn import Dropout
from torch.nn import Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class EfficientFaceModel(EfficientNet):    # just for EfficientNet_b5
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)

        self.output_layer = Sequential(BatchNorm2d(2048),
                                       Dropout(),
                                       Flatten(),
                                       Linear(2048 * 3 * 3, 512),
                                       BatchNorm1d(512))

        self._avg_pooling = None
        self._dropout = None
        self._fc = None

    def forward(self, inputs):
        # Convolution layers
        x = self.extract_features(inputs)
        # print(x.shape)
        x = self.output_layer(x)
        # print(x.shape)

        return x
