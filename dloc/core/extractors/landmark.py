#!/usr/bin/env python
"""
@File    :   landmark.py
@Time    :   2021/06/28 11:07:03
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import cv2
import numpy as np
import torch

from ..utils.base_model import BaseModel


class Landmark(BaseModel):
    default_conf = {
        'sift': False,
    }
    required_inputs = ['image']

    def _init(self, conf, model_path):
        # using sift keypoints
        self.conf = {**self.default_conf, **conf}

        self.with_sift = self.conf['sift']
        if self.with_sift:
            self.sift = cv2.xfeatures2d.SIFT_create()

    def _forward(self, data):
        if self.with_sift:
            kpts = self.sift.detect(data['image'])
            kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
            coord = torch.from_numpy(kpts).float()
            return {
                'keypoints': [coord],
            }
        else:
            return
