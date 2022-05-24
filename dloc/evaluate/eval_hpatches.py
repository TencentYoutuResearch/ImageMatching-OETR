#!/usr/bin/env python
"""
@File    :   eval_imc.py
@Time    :   2021/06/17 17:03:09
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import argparse
import os
from parser import arg_parse

import numpy as np
import torch
from dataloader.hpatches import HpatchesDataset
from tqdm import tqdm
from utils.evaluation import h_evaluate
from utils.utils import get_logger
from utils.visualization import plot_hpatches_mAA


def summary(stats, n_i=52, n_v=56):
    seq_type, n_feats, n_matches = stats
    print('# Features: {:2f} - [{:d}, {:d}]'.format(np.mean(n_feats),
                                                    np.min(n_feats),
                                                    np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((n_i + n_v) * 5),
        np.sum(n_matches[seq_type == 'i']) / (n_i * 5),
        np.sum(n_matches[seq_type == 'v']) / (n_v * 5),
    ))


def log_summary(error, method, logger, n_i=52, n_v=56):
    """_summary_

    Args:
        error (_type_): _description_
        method (_type_): _description_
        logger (_type_): _description_
        n_i (int, optional): _description_. Defaults to 52.
        n_v (int, optional): _description_. Defaults to 56.
    """
    seq_type, n_feats, n_matches = error[-1]

    logger.info(
        '{}\t {:2f} - [{:d}, {:d}]\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            method.ljust(20),
            np.mean(n_feats),
            np.min(n_feats),
            np.max(n_feats),
            np.sum(n_matches) / ((n_i + n_v) * 5),
            np.sum(n_matches[seq_type == 'i']) / (n_i * 5),
            np.sum(n_matches[seq_type == 'v']) / (n_v * 5),
        ))


def benchmark_features(input_pairs,
                       results_path,
                       dataset_path,
                       pairwise=False):
    """_summary_

    Args:
        input_pairs (_type_): _description_
        results_path (_type_): _description_
        dataset_path (_type_): _description_
        pairwise (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    n_feats = []
    n_matches = []
    seq_type = []
    rng = np.arange(1, 16)
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    loader = HpatchesDataset(input_pairs, dataset_path, results_path, pairwise)
    loader = torch.utils.data.DataLoader(loader, num_workers=1)

    for _, data in tqdm(enumerate(loader), total=len(loader)):
        dist = h_evaluate(
            data['H_gt'][0],
            data['kpts0'][0],
            data['kpts1'][0],
            data['matches'][0].transpose(1, 0),
        )
        if dist.shape[0] == 0:
            dist = np.array([float('inf')])

        for thr in rng:
            if data['seq_name'][0][0] == 'i':
                i_err[thr] += np.mean(dist <= thr)
            else:
                v_err[thr] += np.mean(dist <= thr)
        seq_type.append(data['seq_name'][0][0])
        n_matches.append(data['matches'][0].shape[1])
        n_feats.append(data['kpts1'][0].shape[0])

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, [seq_type, n_feats, n_matches]


def main(input_pairs, results_path, methods_file, dataset_path='', viz=False):
    with open(methods_file, 'r') as f:
        methods = [line.split() for line in f.readlines()]
    errors = {}

    for i in range(len(methods)):
        if i == 0:
            logger = get_logger('hpatches.log')
            logger.info(
                'methods\t\t\t Features\t Overall\t Illumination\t Viewpoint\t'
            )
        method = methods[i][1]
        folder = methods[i][0]
        if os.path.exists(os.path.join(results_path, folder)):
            if 'loftr' in method.lower():
                errors[method] = benchmark_features(
                    input_pairs,
                    os.path.join(results_path, folder),
                    dataset_path,
                    pairwise=True,
                )
            else:
                errors[method] = benchmark_features(
                    input_pairs, os.path.join(results_path, folder),
                    dataset_path)
            log_summary(errors[method], method, logger)
        # summary(errors[method][-1])
    if viz:
        plot_hpatches_mAA(errors,
                          np.array(methods)[:, 1],
                          np.array(methods)[:, 1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with HPatches',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser = arg_parse(parser)
    args = parser.parse_args()
    main(**args.__dict__)
