# -*- coding: utf-8 -*-
"""
Author:
    mumuychen@tencent.com

"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

from ..utils.base_model import BaseModel

contextdesc_path = Path(__file__).parent / '../../models/contextdesc'
# model_path = Path(__file__).parent / '../../../weights/'
sys.path.append(str(contextdesc_path))
from models import get_model


class ContextDesc(BaseModel):
    default_conf = {
        'loc_model': 'loc.pb',
        'dense_desc': False,
        'ratio_test': False,
        'n_sample': 2048,
        'n_feature': 10000,
    }
    required_inputs = ['image']

    def _init(self, conf, model_path):
        self.loc_model_path = str(model_path / Path(conf['loc_model']))

        self.local_model = get_model('loc_model')(self.loc_model_path, **{'sift_desc': True,
                                                  'n_sample': conf['n_sample'],
                                                  'peak_thld': 0.04,
                                                  'dense_desc': conf['dense_desc'],
                                                  'upright': False})

    def _forward(self, data):
        gray_img = data['image'][0].cpu().numpy()*255
        gray_img = gray_img.transpose(1,2,0)
        loc_feat, kpt_mb, normalized_xy, cv_kpts, sift_desc =\
                 self.local_model.run_test_data(gray_img)
        if sift_desc is None:
            return {}
        keypoints = np.array([[pt.pt[0], pt.pt[1]] for pt in cv_kpts])
        scores = np.array([pt.response for pt in cv_kpts])
        descriptors = loc_feat / np.linalg.norm(loc_feat, axis=-1, keepdims=True)
        return {
            'keypoints': [torch.from_numpy(keypoints)],
            'scores': [torch.from_numpy(scores)],
            'descriptors': [torch.from_numpy(descriptors.T)],
        }
        
