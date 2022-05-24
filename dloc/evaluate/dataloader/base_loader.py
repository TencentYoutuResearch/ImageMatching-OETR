#!/usr/bin/env python
"""
@File    :   base_loader.py
@Time    :   2022/03/24 11:47:17
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import os

import h5py
import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_pairs,
        results_path=None,
        keypoints=None,
        matches=None,
        pairwise=False,
    ):
        with open(input_pairs, 'r') as f:
            self.pairs_list = [
                line.strip('\n').split(' ') for line in f.readlines()
            ]
        self.seq_name = ''
        self.pairwise = pairwise
        self.results_path = results_path
        self.all_keypoints = keypoints
        self.all_matches = matches

    def __len__(self):
        return len(self.pairs_list)

    def __pairs__(self):
        return self.pairs_list

    def process_data(self, info, seq_name):
        if seq_name != self.seq_name:
            if self.results_path is None:
                self.keypoints = self.all_keypoints[seq_name]
                self.matches = self.all_matches[seq_name]
            else:
                self.keypoints = h5py.File(
                    os.path.join(self.results_path, seq_name, 'keypoints.h5'),
                    'r')
                self.matches = h5py.File(
                    os.path.join(self.results_path, seq_name, 'matches.h5'),
                    'r')
                inparams_path = os.path.join(self.results_path, seq_name,
                                             'inparams.h5')
                if os.path.exists(inparams_path):
                    self.inparams = h5py.File(inparams_path, 'r')
                else:
                    self.inparams = None
                scale_path = os.path.join(self.results_path, seq_name,
                                          'scales.h5')
                if os.path.exists(scale_path):
                    self.scale_diff = h5py.File(scale_path, 'r')
                else:
                    self.scale_diff = None
            self.seq_name = seq_name

        if self.pairwise:
            kpts0 = self.keypoints['{}-{}'.format(
                info[0].split('/')[-1][:-4],
                info[1].split('/')[-1][:-4])].__array__()
            kpts1 = self.keypoints['{}-{}'.format(
                info[1].split('/')[-1][:-4],
                info[0].split('/')[-1][:-4])].__array__()
            if self.inparams is not None and len(self.inparams.keys()) > 0:
                inparams0 = self.inparams['{}-{}'.format(
                    info[0].split('/')[-1][:-4],
                    info[1].split('/')[-1][:-4])].__array__()
                inparams1 = self.inparams['{}-{}'.format(
                    info[1].split('/')[-1][:-4],
                    info[0].split('/')[-1][:-4])].__array__()
            else:
                inparams0 = inparams1 = None
            if self.scale_diff is not None and len(self.scale_diff.keys()) > 0:
                scale_diff = self.scale_diff['{}-{}'.format(
                    info[0].split('/')[-1][:-4],
                    info[1].split('/')[-1][:-4])].__array__()
                scale_diff = float(scale_diff)
            else:
                scale_diff = 1.0
        else:
            kpts0 = self.keypoints[info[0].split('/')[-1][:-4]].__array__()
            kpts1 = self.keypoints[info[1].split('/')[-1][:-4]].__array__()
            inparams0 = inparams1 = None
            scale_diff = 1.0

        matches = self.matches['{}-{}'.format(
            info[0].split('/')[-1][:-4],
            info[1].split('/')[-1][:-4])].__array__()
        return kpts0, kpts1, matches, inparams0, inparams1, scale_diff

    def __getitem__(self, idx):
        pass
