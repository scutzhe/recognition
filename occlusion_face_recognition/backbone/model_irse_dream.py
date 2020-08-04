#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : model_irse_dream.py
# @time    : 8/4/20 4:38 PM
# @desc    : 
'''

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d
from torch.nn import PReLU, ReLU, Sigmoid, Dropout, MaxPool2d
from torch.nn import AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple
from torch.nn.parameter import Parameter

import math
import pickle

# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class ECAModule(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        https://github.com/BangguWu/ECANet/issues/24
    """
    def __init__(self, channel, gamma=2, b=1, k_size=None):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if k_size is None:   # auto select k_size by input channel
            t = int((math.log(channel, 2) + b) / gamma)
            k_size = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                                             BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                                             BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_ECA(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_ECA, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                                             BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            ECAModule(depth, k_size=3)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, end2end, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se', 'ir_eca'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        elif mode == 'ir_eca':
            unit_module = bottleneck_IR_ECA
        else:
            raise Exception(f"not support this model:{mode}")

        self.end2end = end2end
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        ## add stich module
        self.dream = Sequential(Linear(512, 512),
                                ReLU(inplace=True),
                                Linear(512, 512),
                                ReLU(inplace=True))

        self._initialize_weights()


    def forward(self, x, yaw):
        x = self.input_layer(x)
        x = self.body(x)
        mid_feature = self.output_layer(x)
        if self.end2end:
            raw_feature = self.dream(mid_feature)  # raw_feature.size()=(32,512)
            yawTmp = yaw.view(yaw.size(0), 1)  # yaw.size() = (32,512)
            yaw_ = yawTmp.expand_as(raw_feature)  # yaw.size()=(32,512)
            feature = yaw_ * raw_feature + mid_feature  # feature.size()=(32,512)
        else:
            feature = mid_feature
        return feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class LevelAttentionModel(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(LevelAttentionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(feature_size, 1, kernel_size=3, padding=1)

        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.conv5(out)
        out_attention = self.output_act(out)

        return out_attention


class FAN_Backbone(Backbone):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(FAN_Backbone, self).__init__(input_size, num_layers, mode=mode)
        self.levelattention = LevelAttentionModel(num_features_in=512, feature_size=256)

        self.levelattention.conv5.weight.data.fill_(0)
        self.levelattention.conv5.bias.data.fill_(0)

        self.no_attention = False

    def forward(self, inputs):
        x = self.input_layer(inputs)
        feature = self.body(x)
        attention_mask = self.levelattention(feature)

        if self.no_attention is True:     # stage 1:just train out_put layer, no levelattention layer
            out_feature = self.output_layer(feature)
        else:
            attention_feature = torch.exp(attention_mask) * feature
            out_feature = self.output_layer(attention_feature)

        if self.training is True:
            return attention_mask, out_feature
        else:
            return out_feature


class PW_Backbone(Backbone):
    def __init__(self, input_size, num_layers, mode='ir', masksample=None):
        super(PW_Backbone, self).__init__(input_size, num_layers, mode=mode)
        self.levelattention = LevelAttentionModel(num_features_in=512, feature_size=256)

        self.levelattention.conv5.weight.data.fill_(0)
        self.levelattention.conv5.bias.data.fill_(0)

        self.no_attention = False
        if masksample is None:
            self.masksample = None
        else:
            f = open(masksample, 'rb')
            self.masksample = torch.Tensor(pickle.load(f)).cuda()
            f.close()

        # model work type
        self.tpye_train = 0
        self.tpye_valid = 1
        self.tpye_regis = 2
        self.tpye_recog = 3

    def forward(self, inputs, worktype=1):
        if worktype == self.tpye_train:
            occ_inputs, clean_inputs = inputs
            occ_x = self.input_layer(occ_inputs)
            occ_feature = self.body(occ_x)
            occ_attention_mask = self.levelattention(occ_feature)

            clean_x = self.input_layer(clean_inputs)
            clean_feature = self.body(clean_x)
            clean_attention_mask = self.levelattention(clean_feature)

            if self.no_attention is True:     # stage 1:just train out_put layer, no levelattention layer
                occ_feature = self.output_layer(occ_feature)
                clean_feature = self.output_layer(clean_feature)
            else:
                occ_attention_feature = occ_attention_mask.round() * occ_feature
                occ_feature = self.output_layer(occ_attention_feature)

                # use the pairwise occlusion image's mask to occlude the clean feature
                clean_attention_feature = occ_attention_mask.round() * clean_feature
                clean_feature = self.output_layer(clean_attention_feature)

            return occ_attention_mask, occ_feature, clean_attention_mask, clean_feature
        elif worktype == self.tpye_valid:
            x = self.input_layer(inputs)
            feature = self.body(x)
            if self.masksample is None:
                attention_mask = self.levelattention(feature)

                if self.no_attention is True:     # stage 1:just train out_put layer, no levelattention layer
                    feature = self.output_layer(feature)
                else:
                    attention_feature = attention_mask.round() * feature
                    feature = self.output_layer(attention_feature)

                return feature
        elif worktype == self.tpye_regis:
            x = self.input_layer(inputs)
            feature = self.body(x)
            # attention_mask = self.levelattention(feature)
            features = []
            for one_mask in self.masksample:
                attention_feature = one_mask * feature
                features.append(self.output_layer(attention_feature))

            return feature
        elif worktype == self.tpye_recog:
            x = self.input_layer(inputs)
            feature = self.body(x)
            attention_mask = self.levelattention(feature)

            diff = abs(self.masksample.unsqueeze(dim=0) - attention_mask.unsqueeze(dim=1)).sum(dim=[2, 3])
            mask_index = diff.argmin(dim=1)
            min_mask_sample = self.masksample[mask_index]

            attention_feature = min_mask_sample * feature
            feature = self.output_layer(attention_feature)

            return feature, mask_index

        else:
            raise Exception("{} is know work type".format(worktype))


def IR_50(input_size):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model


def IR_101(input_size):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, True, 'ir')

    return model


def IR_152(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_ECA_50(input_size):
    """Constructs a ir_eca-50 model.
    """
    model = Backbone(input_size, 50, 'ir_eca')

    return model


def IR_ECA_101(input_size):
    """Constructs a ir_eca-101 model.
    """
    model = Backbone(input_size, 100, 'ir_eca')

    return model


def IR_ECA_152(input_size):
    """Constructs a ir_eca-152 model.
    """
    model = Backbone(input_size, 152, 'ir_eca')

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model


def FAN_IR_SE_50(input_size):
    model = FAN_Backbone(input_size, 50, 'ir_se')

    return model


def FAN_IR_SE_101(input_size):
    model = FAN_Backbone(input_size, 100, 'ir_se')

    return model


def PW_IR_SE_101(input_size, masksample=None):
    model = PW_Backbone(input_size, 100, 'ir_se', masksample=masksample)

    return model

