#!/usr/bin/env python
"""
@File    :   hpatches.py
@Time    :   2021/06/18 17:37:26
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import os

import h5py
import numpy as np
import torch


class HpatchesDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
    }

    def __init__(self,
                 input_pairs,
                 dataset_path,
                 results_path,
                 pairwise=False):
        with open(input_pairs, 'r') as f:
            self.pairs_list = [line.split() for line in f.readlines()]
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.seq_name = ''
        self.pairwise = pairwise

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        img0, img1 = self.pairs_list[idx]
        seq_name = img1.split('/')[-2]
        im_idx = img1[-5]

        if seq_name != self.seq_name:
            self.keypoints = h5py.File(
                os.path.join(self.results_path, seq_name, 'keypoints.h5'), 'r')
            self.matches = h5py.File(
                os.path.join(self.results_path, seq_name, 'matches.h5'), 'r')
            self.seq_name = seq_name

        if self.pairwise:
            kpts0, kpts1 = self.keypoints['{}-{}'.format(
                img0.split('/')[-1][-5],
                img1.split('/')[-1][-5])].__array__()
        else:
            kpts0 = self.keypoints[img0.split('/')[-1][-5]].__array__()
            kpts1 = self.keypoints[img1.split('/')[-1][-5]].__array__()

        matches = self.matches['{}-{}'.format(
            img0.split('/')[-1][-5],
            img1.split('/')[-1][-5])].__array__()
        H_gt = np.loadtxt(
            os.path.join(self.dataset_path, seq_name, 'H_1_' + im_idx))

        return {
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches': matches,
            'H_gt': H_gt,
            'seq_name': seq_name,
        }


def generate_pairs(root):
    f = open('hpatches.txt', 'w')
    for folder in os.listdir(root):
        for i in range(2, 7):
            f.write('{}/1.ppm {}/{}.ppm\n'.format(folder, folder, i))
