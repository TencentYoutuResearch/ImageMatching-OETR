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
from collections import defaultdict
from parser import arg_parse

import numpy as np
import torch
from dataloader.imc import IMCDataset
from tqdm import tqdm
from utils.evaluation import validation_error
from utils.utils import get_logger, pose_auc, pose_mAA


def summary(state):
    """summary results."""
    print('methods\t\t\t datasets\t AUC@5\t AUC@10\t AUC@20\t',
          'Prec\t MScore\t mAA@10\t')
    for k, v in state.items():
        data, aucs, prec, ms, mAA = v
        for i in range(len(data)):
            name = ''.ljust(20) if i > 0 else k.ljust(20)
            print('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.6f}\t'.
                  format(
                      name,
                      data[i].split('-')[0],
                      aucs[i][0],
                      aucs[i][1],
                      aucs[i][2],
                      prec[i],
                      ms[i],
                      mAA[i],
                  ))


def log_summary(error, method, logger):
    """summary results with logger."""
    data, aucs, prec, ms, mAA = error
    for i in range(len(data)):
        name = ''.ljust(20) if i > 0 else method.ljust(20)
        logger.info(
            '{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.6f}\t'.format(
                name,
                data[i].split('-')[0],
                aucs[i][0],
                aucs[i][1],
                aucs[i][2],
                prec[i],
                ms[i],
                mAA[i],
            ))


def benchmark_features(input_pairs,
                       results_path,
                       pairwise=False,
                       dynamic_threshold='',
                       specific_dataset=''):
    """IMC evaluation implementation.

    Args:
        input_pairs (str): groundtruth pairs information txt file
        results_path (str): result folder with many scenes subfolders
        pairwise (bool, optional): results with pairwise keys. Defaults to False.
        dynamic_threshold (bool, optional): using dynamic threshold for different dataset. Defaults to False.
    """
    loader = IMCDataset(input_pairs,
                        results_path,
                        pairwise=pairwise,
                        specific_dataset=specific_dataset)
    loader = torch.utils.data.DataLoader(loader, num_workers=0)
    pose_errors = defaultdict(list)
    precisions = defaultdict(list)
    matching_scores = defaultdict(list)
    sg_thresholds = {
        'googleurban-val': 0.5,
        'pragueparks-val': 0.8,
        'phototourism-val': 0.45,
    }
    nn_thresholds = {
        'googleurban-val': 0.6,
        'pragueparks-val': 0.75,
        'phototourism-val': 0.6,
    }
    for _, data in tqdm(enumerate(loader), total=len(loader)):
        if dynamic_threshold == 'sg':
            results = validation_error(data, sg_thresholds[data['data'][0]])
        elif dynamic_threshold == 'nn':
            results = validation_error(data, nn_thresholds[data['data'][0]])
        else:
            results = validation_error(data)
        pose_error = np.maximum(results['error_t'], results['error_R'])
        pose_errors[data['data'][0]].append(pose_error)
        precisions[data['data'][0]].append(results['precision'])
        matching_scores[data['data'][0]].append(results['matching_score'])

    thresholds = [5, 10, 20]
    all_aucs = []
    all_prec = []
    all_ms = []
    all_mAA = []
    all_data = []
    for k in matching_scores.keys():
        all_data.append(k)
        aucs = pose_auc(pose_errors[k], thresholds)
        all_aucs.append([100.0 * yy for yy in aucs])
        all_prec.append(100.0 * np.mean(precisions[k]))
        all_ms.append(100.0 * np.mean(matching_scores[k]))
        all_mAA.append(100.0 * pose_mAA(pose_errors[k]))

    all_data.append('total\t')
    aucs = pose_auc(sum(list(pose_errors.values()), []), thresholds)
    all_aucs.append([100.0 * yy for yy in aucs])
    all_prec.append(100.0 * np.mean(sum(list(precisions.values()), [])))
    all_ms.append(100.0 * np.mean(sum(list(matching_scores.values()), [])))
    all_mAA.append(100.0 * pose_mAA(sum(list(pose_errors.values()), [])))
    return [all_data, all_aucs, all_prec, all_ms, all_mAA]


def main(
    input_pairs,
    results_path,
    methods_file,
    dataset_path='',
    specific_dataset='',
    viz=False,
):
    with open(methods_file, 'r') as f:
        methods = [line.split() for line in f.readlines()]
    errors = {}
    for i in range(len(methods)):
        if i == 0:
            logger = get_logger('imc.log')
            logger.info('methods\t\t\t datasets\t AUC@5\t AUC@10\t AUC@20\t'
                        'Prec\t MScore\t mAA@10\t')

        method = methods[i][1]
        folder = methods[i][0]
        dynamic_threshold = ''
        if 'oetr' in method.lower():
            if 'superglue' in method.lower():
                dynamic_threshold = 'sg'
            elif 'nn' in method.lower():
                dynamic_threshold = 'nn'

        if os.path.exists(os.path.join(results_path, folder)):
            if 'loftr' in method.lower() or 'oetr' in method.lower():
                errors[method] = benchmark_features(
                    input_pairs,
                    os.path.join(results_path, folder),
                    pairwise=True,
                    dynamic_threshold=dynamic_threshold,
                    specific_dataset=specific_dataset,
                )
            else:
                errors[method] = benchmark_features(
                    input_pairs,
                    os.path.join(results_path, folder),
                    dynamic_threshold=dynamic_threshold,
                    specific_dataset=specific_dataset,
                )
            log_summary(errors[method], method, logger)
    summary(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with IMC datase',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--specific_dataset',
        type=str,
        default='',
        help='Path to the list of image pairs',
    )
    parser = arg_parse(parser)
    args = parser.parse_args()
    main(**args.__dict__)
