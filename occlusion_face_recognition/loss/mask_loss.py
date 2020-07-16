#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: mask_loss.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2019.11.14 16:25    shengyang      v0.1        creation

import torch.nn as nn


class MaskLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        """
        loss = alpha * occ_part_loss + beta * clean_part_loss
        :param alpha: occlusion part weight
        :param beta: clean part weight
        """
        super(MaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, mask, mask_gt):
        """
        :param mask: a tensor with the value of [0, 1], 0 -> occlusion
        :param mask_gt: the groudtruch with the value of {0,1}, 0 -> occlusion
        :return:
        """

        occ_part = 1 - mask_gt      # occlusion part is 1
        clean_part = mask_gt

        occ_diff = occ_part * mask
        clean_diff = 1 - clean_part * mask
        occ_loss = occ_diff.sum(dim=[1, 2, 3]).mean()
        clean_loss = clean_diff.sum(dim=[1, 2, 3]).mean()

        # loss = self.alpha * occ_loss + self.beta * clean_loss
        # the rate 0~100%
        loss = (self.alpha * occ_loss + self.beta * clean_loss) / 0.49

        return loss
