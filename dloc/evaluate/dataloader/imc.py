#!/usr/bin/env python
"""
@File    :   imc.py
@Time    :   2021/06/21 15:53:56
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import numpy as np

from dloc.evaluate.dataloader.base_loader import BaseDataset


class IMCDataset(BaseDataset):
    def __init__(
        self,
        input_pairs,
        results_path=None,
        keypoints=None,
        matches=None,
        pairwise=False,
        specific_dataset='',
    ):
        if not specific_dataset:
            dataset_list = [
                'googleurban-val', 'pragueparks-val', 'phototourism-val'
            ]
        else:
            dataset_list = [specific_dataset]
        with open(input_pairs, 'r') as f:
            self.pairs_list = [
                line.strip('\n').split(' ') for line in f.readlines()
                if line.split('/')[0] in dataset_list
            ]

        self.results_path = results_path
        self.seq_name = ''
        self.pairwise = pairwise
        self.all_keypoints = keypoints
        self.all_matches = matches

    def __getitem__(self, idx):
        info = self.pairs_list[idx]
        K0 = np.array(info[2:11], dtype=float).reshape(3, 3)
        K1 = np.array(info[11:20], dtype=float).reshape(3, 3)
        pose = np.array(info[20:36], dtype=float).reshape(4, 4)
        seq_name = info[0].split('/')[1]
        data_name = info[0].split('/')[0]
        kpts0, kpts1, matches, inparams0, inparams1, scale_diff = self.process_data(
            info, data_name + '/' + seq_name)

        data = {
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches': matches,
            'intrinsics0': K0,
            'intrinsics1': K1,
            'pose': pose,
            'scene': self.seq_name,
            'data': data_name,
            'name0': info[0],
            'name1': info[1],
            'scale_diff': scale_diff,
        }
        if inparams0 is not None and inparams1 is not None:
            data['inparams0'] = inparams0
            data['inparams1'] = inparams1
        return data
