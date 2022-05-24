#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File    :   cotr.py
@Time    :   2021/06/25 15:04:05
@Author  :   AbyssGaze 
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../models/cotr"))
from COTR.inference.sparse_engine import SparseEngine
from COTR.models import build_model

# model_path = Path(__file__).parent / '../../../weights/'
from COTR.utils import utils


class Cotr(BaseModel):
    default_conf = {
        "weights": "cotr/checkpoint.pth.tar",
        "landmark": False,
        "max_corrs": 300,
        "build": {
            "hidden_dim": 256,
            "dec_layers": 6,
            "dilation": False,
            "layer": "layer3",
            "nheads": 8,
            "position_embedding": "lin_sine",
            "backbone": "resnet50",
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "enc_layers": 6,
        },
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf, model_path, device="cuda"):
        self.conf = {**self.default_conf, **conf}

        args = argparse.Namespace(**self.conf["build"])

        model = build_model(args)
        model = model.cuda()
        model_file = str(model_path / Path(self.conf["weights"]))
        weights = torch.load(model_file, map_location="cpu")["model_state_dict"]
        utils.safe_load_weights(model, weights)
        model = model.eval()
        self.landmark = self.conf["landmark"]

        if self.landmark:
            self.engine = SparseEngine(model, 32, mode="stretching")
        else:
            self.engine = SparseEngine(model, 32, mode="tile")

    def _forward(self, data):
        data["image0"] = data["image0"] * 255
        data["image1"] = data["image1"] * 255
        if "landmark" in data:
            matches = self.engine.cotr_corr_multiscale(
                data["image0"][0].permute(1, 2, 0).cpu().numpy(),
                data["image1"][0].permute(1, 2, 0).cpu().numpy(),
                np.array([1.0]),
                1,
                queries_a=data["landmark"],
                force=False,
            )
        else:
            matches = self.engine.cotr_corr_multiscale_with_cycle_consistency(
                data["image0"][0].permute(1, 2, 0).cpu().numpy(),
                data["image1"][0].permute(1, 2, 0).cpu().numpy(),
                np.linspace(0.5, 0.0625, 4),
                1,
                max_corrs=self.conf["max_corrs"],
                queries_a=None,
            )

        mkpts0 = torch.from_numpy(matches[:, :2]).cuda()
        mkpts1 = torch.from_numpy(matches[:, 2:]).cuda()
        return {
            "keypoints0": [mkpts0],
            "keypoints1": [mkpts1],
            "matches0": [
                torch.from_numpy(np.arange(mkpts0.shape[0])).to(mkpts0.device)
            ],
            "matching_scores0": [torch.ones(mkpts0.shape[0]).to(mkpts0.device)],
        }
