#!/usr/bin/env python
"""
@File    :   megadepth.py
@Time    :   2021/06/18 17:35:23
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import numpy as np

from dloc.evaluate.dataloader.base_loader import BaseDataset


class MegaDepthDataset(BaseDataset):
    """Get MegadepthDataset."""
    def __getitem__(self, idx):
        info = self.pairs_list[idx]
        K0 = np.array(info[2:11], dtype=float).reshape(3, 3)
        K1 = np.array(info[11:20], dtype=float).reshape(3, 3)
        pose = np.array(info[20:36], dtype=float).reshape(4, 4)
        bbox0 = np.array(info[-8:-4], dtype=int)
        bbox1 = np.array(info[-4:], dtype=int)

        seq_name = info[0].split('/')[1]
        data_name = info[0].split('/')[0]
        kpts0, kpts1, matches, inparams0, inparams1, scale_diff = self.process_data(
            info, seq_name)
        pair = '{}-{}'.format(info[0].split('/')[-1][:-4],
                              info[1].split('/')[-1][:-4])
        data = {
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches': matches,
            'intrinsics0': K0,
            'intrinsics1': K1,
            'pose': pose,
            'pair': pair,
            'data': data_name,
            'name0': info[0],
            'name1': info[1],
            'bbox0': bbox0,
            'bbox1': bbox1,
            'scale_diff': scale_diff,
        }
        if inparams0 is not None and inparams1 is not None:
            data['inparams0'] = inparams0
            data['inparams1'] = inparams1
        return data
