#!/usr/bin/env python
"""
@File    :   aslfeat.py
@Time    :   2021/07/20 13:21:52
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import sys
from pathlib import Path

import torch

aslfeat_path = Path(__file__).parent / '../../../third_party/ASLFeat'
sys.path.append(str(aslfeat_path))
from models import get_model  # noqa: E402

from ..utils.base_model import BaseModel  # noqa: E402


class ASLFeat(BaseModel):
    """ASLFeat Convolutional Detector and Descriptor.

    ASLFeat: Learning Local Features of Accurate Shape and Localization.
    Luo, Zixin and Zhou, Lei and Bai, Xuyang and Chen, Hongkai and
    Zhang, Jiahui and Yao, Yao and Li, Shiwei and Fang, Tian and Quan, Long
    In CVPR, 2020. https://arxiv.org/abs/2003.10071
    """

    default_conf = {
        'model': 'model.ckpt-380000',
        'max_keypoints': 2048,
        'net': {
            'max_dim': 2048,
            'config': {
                'kpt_n': 8000,
                'kpt_refinement': True,
                'deform_desc': 1,
                'score_thld': 0.5,
                'edge_thld': 10,
                'multi_scale': True,
                'multi_level': True,
                'nms_size': 3,
                'eof_mask': 5,
                'need_norm': True,
                'use_peakiness': True,
            },
        },
    }
    required_inputs = ['image']

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}
        path = str(model_path / Path('aslfeat/' + self.conf['model']))
        self.model = get_model('feat_model')(path, **self.conf['net'])
        self.topk = conf['max_keypoints']

    def _forward(self, data):
        gray_img = data['image'][0].cpu().numpy() * 255
        gray_img = gray_img.transpose(1, 2, 0)

        descriptors, keypoints, scores = self.model.run_test_data(gray_img)
        idxs = scores.argsort()[-self.topk or None:]
        return {
            'keypoints': [torch.from_numpy(keypoints[idxs])],
            'scores': [torch.from_numpy(scores[idxs])],
            'descriptors': [torch.from_numpy(descriptors[idxs].T)],
        }
