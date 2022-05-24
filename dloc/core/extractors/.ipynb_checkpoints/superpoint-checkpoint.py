import sys
from pathlib import Path

import torch

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../models'))
# model_path = Path(__file__).parent / '../../../weights/'
from SuperGluePretrainedNetwork.superpoint import SuperPoint as SP


class SuperPoint(BaseModel):
    default_conf = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'model_path': 'superpoint_v1.pth'
    }
    required_inputs = ['image']

    def _init(self, conf, model_path, device='cuda'):
        self.conf = {**self.default_conf, **conf}
        self.conf['model_path'] = model_path / self.conf['model_path']
        self.net = SP(self.conf)
        self.net.load_state_dict(torch.load(self.conf['model_path'], map_location=device))
        
    def _forward(self, data):
        return self.net(data)
