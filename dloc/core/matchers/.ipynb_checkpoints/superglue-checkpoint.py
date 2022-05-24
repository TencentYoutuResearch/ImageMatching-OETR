# -*- coding: utf-8 -*-
"""
Author:
    mumuychen@tencent.com

"""

import sys
from pathlib import Path

import torch

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../models'))
# model_path = Path(__file__).parent / '../../../weights/'
from SuperGluePretrainedNetwork.superglue import SuperGlue as SG


class SuperGlue(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'sinkhorn_iterations': 30,
        'match_threshold': 0.2,
        'model_path': 'superglue_outdoor.pth',
    }
    required_inputs = [
        'image0', 'keypoints0', 'scores0', 'descriptors0',
        'image1', 'keypoints1', 'scores1', 'descriptors1',
    ]

    def _init(self, conf, model_path, device='cuda'):
        self.conf = {**self.default_conf, **conf}
        
        self.conf['model_path'] = model_path / 'superglue_{}.pth'.format(self.conf['weights'])
        self.net = SG(self.conf)
        self.net.load_state_dict(torch.load(self.conf['model_path'], map_location=device))

    def _forward(self, data):
        return self.net(data)
