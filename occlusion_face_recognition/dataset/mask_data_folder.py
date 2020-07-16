#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: mask_data_folder.py.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.09.29 16:55    samon      v0.1        creation

from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F

import os
import random
import PIL.Image as Image


class MaskImageFolder(ImageFolder):
    def __init__(self, image_dir, mask_dir, is_valid_file=None, input_size=(112, 112)):
        super(MaskImageFolder, self).__init__(root=image_dir, is_valid_file=is_valid_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.mask_size = (input_size[0] // 16, input_size[1] // 16)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        mask_path = path.replace(self.image_dir, self.mask_dir)
        if os.path.exists(mask_path):
            mask_sample = self.loader(mask_path)
            mask_sample = mask_sample.convert(mode="1")
        else:       # if mask image is not exist, create new mask with same size as sample and fill 1
            mask_sample = Image.new(mode="1", size=sample.size, color=(1,))

        # 1. resize
        sample = sample.resize(size=self.input_size, resample=Image.BICUBIC)
        mask_sample = mask_sample.resize(size=self.mask_size, resample=Image.BICUBIC)

        # 2. random hflip
        if random.random() < 0.5:
            sample = F.hflip(sample)
            mask_sample = F.hflip(mask_sample)

        # 3. to Tensor
        sample = F.to_tensor(sample)
        mask_sample = F.to_tensor(mask_sample)

        # 4. normalize
        sample = F.normalize(sample, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, mask_sample, target
