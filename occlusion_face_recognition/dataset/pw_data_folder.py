#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: pw_data_folder.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.11.14 11:35    shengyang      v0.1        creation
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F

import random
import PIL.Image as Image


class PWImageFolder(ImageFolder):
    def __init__(self, image_dir, mask_dir, is_valid_file=None, input_size=(112, 112)):
        super(PWImageFolder, self).__init__(root=image_dir, is_valid_file=is_valid_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.mask_size = (input_size[0] // 16, input_size[1] // 16)

        self.occ_samples = []
        self.clean_samples = []
        self.label = []
        for path, target in self.samples:
            if path.endswith("_occ.jpg"):
                clean_path = path.replace("_occ.jpg", ".jpg")
                self.occ_samples.append(path)
                self.clean_samples.append(clean_path)
                self.label.append(target)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        occ_path = self.occ_samples[index]
        clean_path = self.clean_samples[index]
        label = self.label[index]
        occ_sample = self.loader(occ_path)
        clean_sample = self.loader(clean_path)

        occ_mask_path = occ_path.replace(self.image_dir, self.mask_dir)
        occ_mask = self.loader(occ_mask_path)
        occ_mask = occ_mask.convert(mode="1")
        # if clean mask image is not exist, create new mask with same size as sample and fill 1
        clean_mask = Image.new(mode="1", size=clean_sample.size, color=(1,))

        # 1. resize
        occ_sample = occ_sample.resize(size=self.input_size, resample=Image.BICUBIC)
        occ_mask = occ_mask.resize(size=self.mask_size, resample=Image.BICUBIC)
        clean_sample = clean_sample.resize(size=self.input_size, resample=Image.BICUBIC)
        clean_mask = clean_mask.resize(size=self.mask_size, resample=Image.BICUBIC)

        # 2. random hflip
        if random.random() < 0.5:
            occ_sample = F.hflip(occ_sample)
            occ_mask = F.hflip(occ_mask)
            clean_sample = F.hflip(clean_sample)
            clean_mask = F.hflip(clean_mask)

        # 3. to Tensor
        occ_sample = F.to_tensor(occ_sample)
        occ_mask = F.to_tensor(occ_mask)
        clean_sample = F.to_tensor(clean_sample)
        clean_mask = F.to_tensor(clean_mask)

        # 4. normalize
        occ_sample = F.normalize(occ_sample, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        clean_sample = F.normalize(clean_sample, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if self.target_transform is not None:
            label = self.target_transform(label)

        # return sample, mask_sample, target
        return occ_sample, occ_mask, clean_sample, clean_mask, label

    def __len__(self):
        return len(self.occ_samples)
