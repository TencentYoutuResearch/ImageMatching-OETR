"""
Author:
    mumuychen@tencent.com

"""
import sys
from pathlib import Path

import torch

disk_path = Path(__file__).parent / '../../../third_party/disk'
sys.path.append(str(disk_path))
from disk.model.disk import DISK  # noqa: E402

from ..utils.base_model import BaseModel  # noqa: E402


class DiskDesc(BaseModel):
    """DISK Convolutional Detector and Descriptor.

    DISK: Learning local features with policy gradient.
    Tyszkiewicz, Micha and Fua, Pascal and Trulls, Eduard.
    In NeurIPS, 2020. https://arxiv.org/abs/2006.13566
    """

    default_conf = {
        'weights': 'depth-save.pth',
        'window': 8,
        'descriptors_dim': 128,
        'kind': 'nms',
        'topk': 1024,
    }
    required_inputs = ['image']

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        model_file = str(model_path / Path('disk') /
                         Path(self.conf['weights']))
        self.net = DISK(window=self.conf['window'],
                        desc_dim=self.conf['descriptors_dim'])
        self.net.load_state_dict(torch.load(model_file)['extractor'])

    def _forward(self, data):
        image = data['image']
        if self.conf['kind'] == 'nms':
            features = self.net.features(image,
                                         kind='nms',
                                         window_size=5,
                                         cutoff=None,
                                         n=self.conf['topk'])
        else:
            features = self.net.features(image, kind='rng')

        return {
            'keypoints': [features[0].kp],
            'scores': [features[0].kp_logp],
            'descriptors': [features[0].desc.T],
        }
