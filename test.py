#!/usr/bin/env python
"""
@File    :   train.py
@Time    :   2021/06/29 17:19:36
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from src.config.default import get_cfg_defaults
from src.model import OETR
from src.utils.utils import read_image, visualize_overlap, visualize_overlap_gt

torch.set_grad_enabled(False)


# test model with image pairs dataset
def main(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config_path)
    model = OETR(cfg.OETR).eval().to(device)
    model.load_state_dict(torch.load(opt.checkpoint))
    with open(opt.input_pairs, 'r') as f:
        pairs = [line.split() for line in f.readlines()]

    for i, pair in enumerate(pairs):
        name1, name2 = pair[:2]
        # Load the image pair.
        image1, inp1, scales1 = read_image(os.path.join(opt.input_dir,
                                                        name1), device,
                                           opt.resize, 0, opt.resize_float)
        image2, inp2, scales2 = read_image(os.path.join(opt.input_dir,
                                                        name2), device,
                                           opt.resize, 0, opt.resize_float)
        # Inference model
        box1, box2 = model.forward_dummy(inp1, inp2)
        output = os.path.join(opt.output_dir, name1 + '-' + name2)
        np_box1 = box1[0].cpu().numpy().astype(int)
        np_box2 = box2[0].cpu().numpy().astype(int)

        if len(pair) > 2:
            # Visualization groundtruth and calculated result
            gt_box1 = np.array(pair[2:6]).astype(int)
            gt_box2 = np.array(pair[6:10]).astype(int)
            visualize_overlap_gt(image1, np_box1, gt_box1, image2, np_box2,
                                 gt_box2, output)
        else:
            # Visualization results
            visualize_overlap(image1, np_box1, image2, np_box2, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate megadepth image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--input_pairs',
        type=str,
        default='assets/megadepth/pairs.txt',
        help='Path to the list of image pairs',
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='assets/megadepth/',
        help='Path to the directory that contains the images',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/',
        help='Path to the directory that contains the images',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='assets/checkpoints/models.pth',
        help='Path to the checkpoints of matching model',
    )
    parser.add_argument('--num_layers',
                        type=int,
                        default=50,
                        help='resnet layers')
    parser.add_argument(
        '--resize',
        type=int,
        nargs='+',
        default=[640, 480],
        help='Resize the input image before running inference. If two numbers,'
        'resize to the exact dimensions, if one number, resize the max '
        'dimension, if -1, do not resize',
    )
    parser.add_argument(
        '--resize_float',
        action='store_true',
        help='Resize the image after casting uint8 to float',
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='configs/oetr_config.py',
        help='Path to the configuration of model',
    )
    opt = parser.parse_args()
    Path(opt.output_dir).mkdir(exist_ok=True, parents=True)
    main(opt)
