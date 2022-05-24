#!/usr/bin/env python
"""
@File    :   superpoint.py
@Time    :   2021/04/28 11:20:03
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).parent /
        '../../../third_party/SuperGluePretrainedNetwork'))

from models.superpoint import SuperPoint as SP  # noqa: E402

from ..utils.base_model import BaseModel  # noqa: E402


class SuperPoint(BaseModel):
    """SuperPoint Convolutional Detector and Descriptor.

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
    """

    default_conf = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'model_path': 'superpoint_v1.pth',
    }
    required_inputs = ['image']

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}
        self.conf['model_path'] = model_path / self.conf['model_path']
        self.net = SP(self.conf)

    def _forward(self, data):
        return self.net(data)
