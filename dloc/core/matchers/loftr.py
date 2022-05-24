#!/usr/bin/env python
"""
@File    :   loftr.py
@Time    :   2021/06/28 14:53:53
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent / '../../../third_party//'))

from LoFTR.src.loftr.loftr import LoFTR  # noqa: E402
from LoFTR.src.loftr.utils.cvpr_ds_config import default_cfg  # noqa: E402

from ..utils.base_model import BaseModel  # noqa: E402


class loftr(BaseModel):
    """COTR Convolutional Detector and Matcher.

    LoFTR: Detector-Free Local Feature Matching with Transformers.
    Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei.
    In CVPR, 2021. https://arxiv.org/abs/2104.00680
    """

    default_conf = {
        'weights': 'loftr/outdoor_ds.ckpt',
    }
    required_inputs = [
        'image0',
        'image1',
    ]

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}
        self.model = LoFTR(config=default_cfg)
        self.model.load_state_dict(
            torch.load(os.path.join(model_path,
                                    self.conf['weights']))['state_dict'])
        self.model = self.model.eval().cuda()

    def _forward(self, data):
        batch = {'image0': data['image0'], 'image1': data['image1']}
        self.model(batch)
        mkpts0 = batch['mkpts0_f']
        mkpts1 = batch['mkpts1_f']
        mconf = batch['mconf']
        matches = torch.from_numpy(np.arange(mkpts0.shape[0])).to(
            mkpts0.device)
        return {
            'keypoints0': [mkpts0],
            'keypoints1': [mkpts1],
            'matches0': [matches],
            'matching_scores0': [mconf],
        }
