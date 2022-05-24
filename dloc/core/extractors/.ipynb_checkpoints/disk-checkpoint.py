# -*- coding: utf-8 -*-
"""
Author:
    mumuychen@tencent.com

"""
import sys
from pathlib import Path

import torch

from ..utils.base_model import BaseModel

disk_path = Path(__file__).parent / '../../models/'
sys.path.append(str(disk_path))

from disk.model.disk import DISK


class DiskDesc(BaseModel):
    default_conf = {
        'weights': 'depth-save.pth',
        'window': 8,
        'descriptors_dim': 128,
        'kind': 'nms',
        'topk': 1024
    }
    required_inputs = ['image']

    def _init(self, conf, model_path, device='cuda'):
        self.conf = {**self.default_conf, **conf}
        
        model_file = str(model_path /Path('disk')/ Path(self.conf['weights']))
        self.net = DISK(window = self.conf['window'], desc_dim = self.conf['descriptors_dim'], 
                        with_swin = 'swin' in self.conf['weights'])
        if 'swin' in self.conf['weights']:
            self.net.load_state_dict(torch.load(model_file, map_location=device)['disk'])
        else:
            self.net.load_state_dict(torch.load(model_file, map_location=device)['extractor'])

    def _forward(self, data):
        image = data['image']
        if self.conf['kind'] == 'nms':
            features = self.net.features(image, kind = 'nms', window_size = 5, 
                                    cutoff=None, n = self.conf['topk'])
        else:
            features = self.net.features(image, kind = 'rng')

        return {
            'keypoints': [features[0].kp],
            'scores': [features[0].kp_logp],
            'descriptors': [features[0].desc.T],
        }
        
