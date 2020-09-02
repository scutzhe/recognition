#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: tmp.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.11.14 16:39    shengyang      v0.1        creation

import PIL.Image as Image
import torch
import torch.nn.functional as F
import numpy as np
import pickle


mask = [[[1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 0, 0, 1, 1]],

        [[1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 1, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1]],

        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1]],

        [[1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1, 1, 1]],

        [[1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1]],

        [[1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 1, 1],
         [1, 0, 0, 0, 0, 1, 1]],

        [[1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1]],

        [[1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1]],

        [[1, 1, 0, 0, 0, 0, 1],
         [1, 1, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1]],

        [[1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1]]]

mask_sample = torch.Tensor(mask)

masks = torch.Tensor(np.random.random((1, 7, 7)))
# masks = torch.Tensor(np.ones((4, 7, 7)))

print(mask_sample.unsqueeze(dim=0).size())
print(masks.unsqueeze(dim=1).size())
diff = abs(mask_sample.unsqueeze(dim=0) - masks.unsqueeze(dim=1)).sum(dim=[2, 3])
print(diff)
min_index = diff.argmin(dim=1)
print(min_index)
min_mask_sample = mask_sample[min_index]

print(min_mask_sample.size())

# for one_mask_sample in min_mask_sample:
#     print(one_mask_sample)

# with open("mask_sample.pkl", 'wb') as f:
#     pickle.dump(mask, f)
with open("mask_sample.pkl", 'rb') as f:
    mask = pickle.load(f)
    print(mask.shape)
