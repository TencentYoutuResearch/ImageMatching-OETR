#!/usr/bin/env python
"""
@File    :   argparse.py
@Time    :   2021/06/17 14:03:09
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""


# Evaluation arguments for all datasets
def arg_parse(parser):
    parser.add_argument(
        '--input_pairs',
        type=str,
        default='assets/haptches/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs',
    )
    parser.add_argument(
        '--results_path',
        type=str,
        default='assets/haptches/results',
        help='Path to the list of image pairs',
    )
    parser.add_argument(
        '--methods_file',
        type=str,
        default='assets/haptches/methods.txt',
        help='Path to the list of image pairs',
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='assets/haptches/images',
        help='Path to the list of image pairs',
    )
    parser.add_argument('--viz',
                        action='store_true',
                        help='Visualization of mAA curve.')
    return parser
