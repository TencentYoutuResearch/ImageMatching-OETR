#!/usr/bin/env python
"""
@File    :   imc.py
@Time    :   2021/06/22 10:55:10
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm


def calib_to_matrix(calib):
    """calib Dict to 3x3 matrix."""
    pose = np.empty((4, 4))
    pose[:3, :3] = calib['R'].__array__()
    pose[:3, 3] = calib['T'].__array__()
    pose[3, :] = [0, 0, 0, 1]
    return pose


def generate_pairs(scenes_path, datasets, overlap_ratio=0.1):
    """From origin dataset generate image pairs."""
    pair_file = open('imc_{}.txt'.format(overlap_ratio), 'w')

    with open(scenes_path, 'r') as f:
        scenes_info = [line.strip('\n').split(' ') for line in f.readlines()]

    for info in tqdm(scenes_info, total=len(scenes_info)):
        scene, suffix = info[0], info[1]
        pairs_info = np.load(
            os.path.join(
                datasets,
                scene,
                'set_100/new-vis-pairs/keys-th-{}.npy'.format(overlap_ratio),
            ))

        for i in range(len(pairs_info)):
            name0, name1 = pairs_info[i].split('-')
            calib0 = h5py.File(
                os.path.join(
                    datasets,
                    scene,
                    'set_100/calibration/calibration_{}.h5'.format(name0),
                ),
                'r',
            )
            calib1 = h5py.File(
                os.path.join(
                    datasets,
                    scene,
                    'set_100/calibration/calibration_{}.h5'.format(name1),
                ),
                'r',
            )
            K0 = ' '.join(map(str, calib0['K'].__array__().reshape(-1)))
            K1 = ' '.join(map(str, calib1['K'].__array__().reshape(-1)))
            pose0 = calib_to_matrix(calib0)
            pose1 = calib_to_matrix(calib1)
            relative_pose = ' '.join(
                map(str, (np.matmul(pose1, np.linalg.inv(pose0)).reshape(-1))))
            pair_file.write(
                '{}/set_100/images/{}.{} {}/set_100/images/{}.{} {} {} {}\n'.
                format(scene, name0, suffix, scene, name1, suffix, K0, K1,
                       relative_pose))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate IMC image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--scenes',
        type=str,
        default='assets/imc/scenes.txt',
        help='Path to the list of scenes',
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='assets/imc/',
        help='Path to the list of image pairs',
    )
    parser.add_argument('--overlap_ratio',
                        type=float,
                        default=0.1,
                        help='overlap_ratio')
    args = parser.parse_args()
    generate_pairs(args.scenes, args.datasets, args.overlap_ratio)
