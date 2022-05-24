#!/usr/bin/env python
"""
@File    :   superglue.py
@Time    :   2021/05/24 16:27:17
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).parent /
        '../../../third_party/SuperGluePretrainedNetwork'))
from models.superglue import SuperGlue as SG  # noqa: E402

from ..utils.base_model import BaseModel  # noqa: E402


class SuperGlue(BaseModel):
    """SuperGlue feature matching middle-end.

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763
    """

    default_conf = {
        'weights': 'outdoor',
        'sinkhorn_iterations': 30,
        'match_threshold': 0.2,
        'model_path': 'superglue_outdoor.pth',
    }
    required_inputs = [
        'image0',
        'keypoints0',
        'scores0',
        'descriptors0',
        'image1',
        'keypoints1',
        'scores1',
        'descriptors1',
    ]

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}

        self.conf['model_path'] = model_path / 'superglue_{}.pth'.format(
            self.conf['weights'])
        self.net = SG(self.conf)

    def _forward(self, data):
        return self.net(data)
