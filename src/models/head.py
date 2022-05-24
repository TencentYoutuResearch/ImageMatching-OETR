#!/usr/bin/env python
'''
@File    :   head.py
@Time    :   2021/07/20 17:07:56
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
'''

import math

import torch
import torch.nn.functional as F
from torch import nn


class DynamicConv(nn.Module):
    def __init__(self, hidden_dim):
        super(DynamicConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        num_output = self.hidden_dim * self.hidden_dim
        self.out_layer = nn.Linear(num_output, self.hidden_dim * 2)
        self.norm2 = nn.LayerNorm(self.hidden_dim * 2)

    def forward(self, features, pro_features):
        '''
        pro_features: (B, W*H, C1)
        feature: (B, H*W, C2)
        '''
        features = torch.bmm(features, pro_features)
        features = self.norm1(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm2(features)
        features = self.activation(features)

        return features


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FCOSHead(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 prior_prob=0.01,
                 stride=16,
                 norm_reg_targets=False,
                 centerness_on_reg=True,
                 training=True):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = 1
        self.stride = stride
        self.training = training
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg

        self.cls_tower = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.GroupNorm(32, in_channels), nn.ReLU())
        self.bbox_tower = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.GroupNorm(32, in_channels), nn.ReLU())

        self.cls_logits = nn.Conv2d(in_channels,
                                    num_classes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.centerness = nn.Conv2d(in_channels,
                                    1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        # initialization
        for modules in [
                self.cls_tower, self.bbox_tower, self.cls_logits,
                self.bbox_pred, self.centerness
        ]:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    torch.nn.init.normal_(module.weight, std=0.01)
                    torch.nn.init.constant_(module.bias, 0)

        # initialize the bias for focal loss

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = Scale(init_value=1.0)

    def forward(self, x):
        cls_tower = self.cls_tower(x)
        box_tower = self.bbox_tower(x)

        logits = self.cls_logits(cls_tower)
        if self.centerness_on_reg:
            centerness = self.centerness(box_tower)
        else:
            centerness = self.centerness(cls_tower)

        bbox_pred = self.scales(self.bbox_pred(box_tower))

        if self.norm_reg_targets:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred = bbox_pred * self.stride
        else:
            bbox_pred = torch.exp(bbox_pred)
        return logits, bbox_pred, centerness
