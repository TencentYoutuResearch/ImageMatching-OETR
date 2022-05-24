#!/usr/bin/env python
"""
@File    :   oetr.py
@Time    :   2021/07/28 17:20:43
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import torch

from dloc.core.utils.base_model import BaseModel  # noqa: E402
from src.config.default import get_cfg_defaults
from src.model import build_detectors


class OETR(BaseModel):
    default_conf = {
        'model': 'oetr',
        'num_layers': 50,
        'stride': 32,
        'last_layer': 1024,
        'weights': 'oetr.pth',
    }
    required_inputs = [
        'image0',
        'image1',
    ]

    def build_cfg(self, conf):
        cfg = get_cfg_defaults()
        cfg.OETR.MODEL = conf['model']
        cfg.OETR.BACKBONE.STRIDE = conf['stride']
        cfg.OETR.BACKBONE.LAYER = conf['layer']
        cfg.OETR.BACKBONE.LAST_LAYER = conf['last_layer']
        return cfg

    def _init(self, conf, model_path):
        # pdb.set_trace()
        self.conf = {**self.default_conf, **conf}
        self.cfg = self.build_cfg(self.conf)
        self.net = build_detectors(self.cfg.OETR)
        model_file = model_path / self.conf['weights']
        self.net.load_state_dict(torch.load(model_file))

    def _forward(self, data):
        box1, box2 = self.net.forward_dummy(data['image0'], data['image1'])
        return box1, box2
