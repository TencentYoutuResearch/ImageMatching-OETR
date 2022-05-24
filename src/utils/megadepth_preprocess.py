#!/usr/bin/env python
"""
@File    :   megadepth.py
@Time    :   2021/06/21 17:13:41
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import argparse
import os
from multiprocessing import Pool

import h5py
import numpy as np

from src.datasets.utils import numpy_overlap_box


def boxes(points):
    box = np.array(
        [points[0].min(), points[1].min(), points[0].max(), points[1].max()])
    return box


# def overlap_box(K1, depth1, pose1, K2, depth2, pose2):
#     mask1 = np.where(depth1 > 0)
#     u1, v1 = mask1[1], mask1[0]
#     Z1 = depth1[v1, u1]

#     # COLMAP convention
#     X1 = (u1 - K1[0, 2]) * (Z1 / K1[0, 0])
#     Y1 = (v1 - K1[1, 2]) * (Z1 / K1[1, 1])
#     XYZ1_hom = np.concatenate(
#         [
#             X1.reshape(1, -1),
#             Y1.reshape(1, -1),
#             Z1.reshape(1, -1),
#             np.ones_like(Z1.reshape(1, -1)),
#         ],
#         axis=0,
#     )
#     XYZ2_hom = pose2 @ np.linalg.inv(pose1) @ XYZ1_hom
#     XYZ2 = XYZ2_hom[:-1, :] / XYZ2_hom[-1, :].reshape(1, -1)

#     uv2_hom = K2 @ XYZ2
#     uv2 = uv2_hom[:-1, :] / uv2_hom[-1, :].reshape(1, -1)
#     h, w = depth2.shape
#     i = uv2[0, :].astype(int)
#     j = uv2[1, :].astype(int)

#     valid_corners = np.logical_and(np.logical_and(i >= 0, j >= 0),
#                                    np.logical_and(i < w, j < h))

#     valid_uv1 = np.stack((u1[valid_corners], v1[valid_corners])).astype(int)
#     valid_uv2 = uv2[:, valid_corners].astype(int)
#     # depth validation
#     Z2 = depth2[valid_uv2[1], valid_uv2[0]]
#     inlier_mask = np.absolute(XYZ2[2, valid_corners] - Z2) < 0.5

#     valid_uv1 = valid_uv1[:, inlier_mask]
#     valid_uv2 = valid_uv2[:, inlier_mask]
#     if valid_uv1.shape[1] == 0 or valid_uv2.shape[1] == 0:
#         return np.array([0] * 4), np.array([0] * 4)

#     box1 = boxes(valid_uv1)
#     box2 = boxes(valid_uv2)
#     return box1, box2


def scale_diff(bbox0, bbox1, depth0, depth1):
    """Calculate the max scale difference in width and height.

    Args:
        bbox0 (np.array): co-visible area bounding box in image0
        bbox1 (np.array): co-visible area bounding box in image1
    Returns:
        scale_dirff(float): the max difference in width and height
    """
    w_diff = max(
        (bbox0[2] - bbox0[0]) / (bbox1[2] - bbox1[0]),
        (bbox1[2] - bbox1[0]) / (bbox0[2] - bbox0[0]),
    )
    h_diff = max(
        (bbox0[3] - bbox0[1]) / (bbox1[3] - bbox1[1]),
        (bbox1[3] - bbox1[1]) / (bbox0[3] - bbox0[1]),
    )
    # image_h_scale = max(depth0.shape[0]/(bbox0[3] - bbox0[1]),
    #                     depth1.shape[0]/(bbox1[3] - bbox1[1]))
    # image_w_scale = max(depth0.shape[1]/(bbox0[2] - bbox0[0]),
    #                     depth1.shape[1]/(bbox1[2] - bbox1[0]))
    return max(w_diff, h_diff)  # image_h_scale, image_w_scale)


def megadepth_callback(data):
    with open('megadepth_train_scale.txt', 'a+') as pair_file:
        pair_file.write(data)


def get_existed_pairs(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        pairs = [line.strip(' ') for line in f.readlines()]
    existed_pairs = []
    for info in pairs:
        pair = info[0] + '-' + info[5]
        existed_pairs.append(pair)
    return existed_pairs


def process_scene(
    scene,
    datasets,
    pairs_per_scene=3000,
    existed_pairs=[],
    min_overlap_ratio=0.1,
    max_overlap_ratio=0.7,
    max_scale_ratio=100,
):
    """Preprocess scene to txt to accelerate training.

    Args:
        scene (str): scene id in megadepth
        datasets (str): path of megadepth dataset
        pairs_per_scene (int, optional): extract the number of image pairs from the data. Defaults to 3000.
        existed_pairs (list, optional): extracted pairs. Defaults to [].
        min_overlap_ratio (float, optional): min overlap ratio. Defaults to 0.1.
        max_overlap_ratio (float, optional): max overlap ratio. Defaults to 0.7.
        max_scale_ratio (int, optional): max scale difference ratio. Defaults to 100.

    Returns:
        str: all pairs information
    """
    data = ''
    scene_info_path = os.path.join(datasets, 'scene_info/%s.0.npz' % scene)
    pairs_repeate = existed_pairs

    if not os.path.exists(scene_info_path):
        return data
    scene_info = np.load(scene_info_path, allow_pickle=True)
    overlap_matrix = scene_info['overlap_matrix']
    scale_ratio_matrix = scene_info['scale_ratio_matrix']

    valid = np.logical_and(
        np.logical_and(
            overlap_matrix >= min_overlap_ratio,
            overlap_matrix <= max_overlap_ratio,
        ),
        scale_ratio_matrix <= max_scale_ratio,
    )
    image_paths = scene_info['image_paths']
    depth_paths = scene_info['depth_paths']
    intrinsics = scene_info['intrinsics']
    poses = scene_info['poses']

    pairs = np.vstack(np.where(valid))
    print('{} total length:{}'.format(scene, pairs.shape[1]))
    selected_ids = np.arange(pairs.shape[1])
    np.random.shuffle(selected_ids)

    valid = 0
    for pair_idx in selected_ids:
        idx0 = pairs[0, pair_idx]
        idx1 = pairs[1, pair_idx]
        pair0 = '{}-{}'.format(image_paths[idx0], image_paths[idx1])
        pair1 = '{}-{}'.format(image_paths[idx1], image_paths[idx0])
        if pair0 in pairs_repeate or pair1 in pairs_repeate:
            continue
        pairs_repeate.append(pair0)

        K0 = intrinsics[idx0]
        K1 = intrinsics[idx1]
        pose0 = poses[idx0]
        pose1 = poses[idx1]

        depth_path0 = os.path.join(datasets, depth_paths[idx0])
        with h5py.File(depth_path0, 'r') as hdf5_file:
            depth0 = np.array(hdf5_file['/depth'])

        depth_path1 = os.path.join(datasets, depth_paths[idx1])
        with h5py.File(depth_path1, 'r') as hdf5_file:
            depth1 = np.array(hdf5_file['/depth'])

        # bbox0, bbox1 = overlap_box(K0, depth0, pose0, K1, depth1, pose1)
        bbox0, _, bbox1, _, bbox_valid = numpy_overlap_box(
            K0,
            depth0,
            pose0,
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            K1,
            depth1,
            pose1,
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
        )

        if bbox_valid and (bbox0.max() > 0 and bbox1.max() > 0
                           and scale_diff(bbox0, bbox1, depth0, depth1) > 2):
            K0 = ','.join(map(str, K0.reshape(-1)))
            K1 = ','.join(map(str, K1.reshape(-1)))
            pose0 = ','.join(map(str, pose0.reshape(-1)))
            pose1 = ','.join(map(str, pose1.reshape(-1)))
            bbox0 = ','.join(map(str, bbox0.reshape(-1)))
            bbox1 = ','.join(map(str, bbox1.reshape(-1)))
            info = '{} {} {} {} {} {} {} {} {} {}\n'.format(
                image_paths[idx0],
                depth_paths[idx0],
                K0,
                pose0,
                bbox0,
                image_paths[idx1],
                depth_paths[idx1],
                K1,
                pose1,
                bbox1,
            )
            data += info
            valid += 1
        if valid == pairs_per_scene:
            return data
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate megadepth image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--scenes',
        type=str,
        default='assets/megadepth/validation.txt',
        help='Path to the list of scenes',
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='assets/megadepth/',
        help='Path to the list of image pairs',
    )
    parser.add_argument(
        '--input_pairs',
        type=str,
        default='assets/megadepth/megadepth_0.4.txt',
        help='Path to the list of image pairs',
    )
    parser.add_argument('--max_overlap_ratio',
                        type=float,
                        default=0.7,
                        help='max_overlap_ratio')
    parser.add_argument('--pairs_per_scene',
                        type=int,
                        default=300,
                        help='pairs_per_scene')

    opt = parser.parse_args()
    with open(opt.scenes, 'r') as f:
        scenes = [line.strip('\n') for line in f.readlines()]
    existed_pairs = get_existed_pairs(opt.input_pairs)
    pool = Pool(processes=6)
    for scene in scenes:
        pool.apply_async(
            process_scene,
            (scene, opt.datasets, opt.pairs_per_scene, existed_pairs),
            callback=megadepth_callback,
        )

    pool.close()
    pool.join()
