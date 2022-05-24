#!/usr/bin/env python
"""
@File    :   preprocess_overlap.py
@Time    :   2021/06/29 11:11:49
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import argparse
import datetime
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.utils import numpy_overlap_box


def visualize_box(image1, bbox1, points1, depth1, image2, bbox2, points2,
                  depth2, output):
    """visualization image pairs co-visible area.

    Args:
        image1, image2 (np.array): origin image
        bbox1, bbox2 (np.array): co-visible bounding box
        points1, points2 (np.array): valid depth 2D uv coordinate
        depth1, depth2 (np.array): depth map of image1 and image2
        output (str): output directory
    """
    # Origin image pairs
    left = cv2.rectangle(
        np.stack([image1.numpy()] * 3, -1)[0], bbox1[0], bbox1[1], (255, 0, 0),
        2)
    right = cv2.rectangle(
        np.stack([image2.numpy()] * 3, -1)[0], bbox2[0], bbox2[1], (0, 0, 255),
        2)
    viz = cv2.hconcat([left, right])

    # Visualize mask
    mask1 = np.zeros((left.shape), dtype=np.float32)
    mask2 = np.zeros((right.shape), dtype=np.float32)
    for i in range(points1.shape[1]):
        mask1 = cv2.circle(mask1, (points1[0, i], points1[1, i]), 1,
                           (255, 0, 0))
    for i in range(points2.shape[1]):
        mask2 = cv2.circle(mask2, (points2[0, i], points2[1, i]), 1,
                           (0, 0, 255))

    left = cv2.addWeighted(left, 0.5, mask1, 0.5, 0)
    right = cv2.addWeighted(right, 0.5, mask2, 0.5, 0)
    viz = cv2.hconcat([left, right])

    # Visualize depth
    depth_viz = cv2.hconcat([
        np.stack([depth1.numpy()] * 3, -1) * 10,
        np.stack([depth2.numpy()] * 3, -1) * 10,
    ])
    all_viz = cv2.vconcat([viz, depth_viz])
    cv2.imwrite('all_' + output, all_viz)


class MegaDepthDataset(Dataset):
    def __init__(
        self,
        scene_list_path='assets/train_scenes_all.txt',
        scene_info_path='assets/megadepth/scene_info',
        base_path='assets/megadepth',
        train=True,
        min_overlap_ratio=0.1,
        max_overlap_ratio=0.7,
        max_scale_ratio=100,
        preprocessing=None,
        pairs_per_scene=1000,
        image_size=[720, 720],
        with_mask=False,
    ):
        self.scenes = []
        with open(scene_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip('\n'))

        self.scene_info_path = scene_info_path
        self.base_path = base_path
        self.preprocessing = preprocessing
        self.train = train

        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_scale_ratio = max_scale_ratio

        self.pairs_per_scene = pairs_per_scene

        self.image_size = image_size
        self.with_mask = with_mask

        self.dataset = []

    def caculate_depth(pos, depth):
        """calculate valid depth of point cloud.

        Args:
            pos (tensor): 2D points cloud with x,y coordinate
            depth (tensor): depth map of scene

        Returns:
            interpolated_depth: depth value
            pos: valid depth position
            ids: valid depth index
        """
        ids = torch.arange(0, pos.size(1))
        h, w = depth.size()
        i = pos[0, :].long()
        j = pos[1, :].long()
        valid_corners = torch.min(torch.min(i >= 0, j >= 0),
                                  torch.min(i < h, j < w))
        ids = ids[valid_corners]
        if ids.size(0) == 0:
            return [ids, ids, ids]
        valid_depth = depth[i[valid_corners], j[valid_corners]] > 0
        ids = ids[valid_depth]
        if ids.size(0) == 0:
            return [ids, ids, ids]
        i = i[ids]
        j = j[ids]
        interpolated_depth = depth[i, j]
        pos = torch.cat(
            [pos[0, :][ids].view(1, -1), pos[1, :][ids].view(1, -1)], dim=0)
        return [interpolated_depth, pos, ids]

    def build_dataset(self):
        self.dataset = []
        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)

        print('{} building datasets...'.format(
            datetime.datetime.now().strftime('%y-%m-%d-%H:%M')))

        for scene in self.scenes:
            scene_info_path = os.path.join(self.scene_info_path,
                                           '%s.0.npz' % scene)
            if not os.path.exists(scene_info_path):
                continue
            scene_info = np.load(scene_info_path, allow_pickle=True)
            overlap_matrix = scene_info['overlap_matrix']
            scale_ratio_matrix = scene_info['scale_ratio_matrix']

            valid = np.logical_and(
                np.logical_and(
                    overlap_matrix >= self.min_overlap_ratio,
                    overlap_matrix <= self.max_overlap_ratio,
                ),
                scale_ratio_matrix <= self.max_scale_ratio,
            )

            pairs = np.vstack(np.where(valid))
            if self.pairs_per_scene:
                selected_ids = np.random.choice(pairs.shape[1],
                                                self.pairs_per_scene)
            else:
                selected_ids = np.arange(pairs.shape[1])

            image_paths = scene_info['image_paths']
            depth_paths = scene_info['depth_paths']
            points3D_id_to_2D = scene_info['points3D_id_to_2D']
            points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']
            intrinsics = scene_info['intrinsics']
            poses = scene_info['poses']

            for pair_idx in selected_ids:
                idx1 = pairs[0, pair_idx]
                idx2 = pairs[1, pair_idx]
                matches = np.array(
                    list(points3D_id_to_2D[idx1].keys()
                         & points3D_id_to_2D[idx2].keys()))

                # Scale filtering
                matches_nd1 = np.array(
                    [points3D_id_to_ndepth[idx1][match] for match in matches])
                matches_nd2 = np.array(
                    [points3D_id_to_ndepth[idx2][match] for match in matches])
                scale_ratio = np.maximum(matches_nd1 / matches_nd2,
                                         matches_nd2 / matches_nd1)
                matches = matches[np.where(
                    scale_ratio <= self.max_scale_ratio)[0]]
                if matches.shape[0] < 10:
                    continue
                points2D_image1 = np.array([[
                    int(points3D_id_to_2D[idx1][idx][0]),
                    int(points3D_id_to_2D[idx1][idx][1]),
                ] for idx in matches])
                points2D_image2 = np.array([[
                    int(points3D_id_to_2D[idx2][idx][0]),
                    int(points3D_id_to_2D[idx2][idx][1]),
                ] for idx in matches])
                point3D_id = np.random.choice(matches)
                point2D1 = points3D_id_to_2D[idx1][point3D_id]
                point2D2 = points3D_id_to_2D[idx2][point3D_id]
                nd1 = points3D_id_to_ndepth[idx1][point3D_id]
                nd2 = points3D_id_to_ndepth[idx2][point3D_id]
                central_match = np.array(
                    [point2D1[1], point2D1[0], point2D2[1], point2D2[0]])
                # match_file_name = image_paths[idx1].split(
                #     '/')[-1] + '_' + image_paths[idx2].split('/')[-1]

                self.dataset.append({
                    'image_path1': image_paths[idx1],
                    'depth_path1': depth_paths[idx1],
                    'intrinsics1': intrinsics[idx1],
                    'pose1': poses[idx1],
                    'image_path2': image_paths[idx2],
                    'depth_path2': depth_paths[idx2],
                    'intrinsics2': intrinsics[idx2],
                    'pose2': poses[idx2],
                    'central_match': central_match,
                    'points2D_image1': points2D_image1,
                    'points2D_image2': points2D_image2,
                    'scale_ratio': max(nd1 / nd2, nd2 / nd1),
                })
        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (
            image1,
            depth1,
            intrinsics1,
            pose1,
            bbox1,
            resize_ratio1,
            bbox1,
            mask1,
            image2,
            depth2,
            intrinsics2,
            pose2,
            bbox2,
            resize_ratio2,
            bbox2,
            mask2,
            match_file_name,
            central_match,
        ) = self.recover_pair(self.dataset[idx])

        return {
            'depth1': torch.from_numpy(depth1.astype(np.float32)),
            'intrinsics1': torch.from_numpy(intrinsics1.astype(np.float32)),
            'pose1': torch.from_numpy(pose1.astype(np.float32)),
            'ratio1': torch.from_numpy(np.asarray(resize_ratio1, np.float32)),
            'bbox1': torch.from_numpy(bbox1.astype(np.float32)),
            'image0': torch.from_numpy(image1 / 255.0).float()[None],
            'image1': torch.from_numpy(image2 / 255.0).float()[None],
            'mask1': torch.from_numpy(mask1.astype(np.uint8)),
            'mask2': torch.from_numpy(mask2.astype(np.uint8)),
            'file_name': match_file_name,
            'depth2': torch.from_numpy(depth2.astype(np.float32)),
            'intrinsics2': torch.from_numpy(intrinsics2.astype(np.float32)),
            'pose2': torch.from_numpy(pose2.astype(np.float32)),
            'ratio2': torch.from_numpy(np.asarray(resize_ratio2, np.float32)),
            'bbox2': torch.from_numpy(bbox2.astype(np.float32)),
            'central_match':
            torch.from_numpy(central_match.astype(np.float32)),
        }


def main(
    scene_list_path,
    scene_info_path,
    dataset_path,
    batch_size,
    num_workers,
    local_rank=0,
):
    dataset = MegaDepthDataset(
        scene_list_path=scene_list_path,
        scene_info_path=scene_info_path,
        base_path=dataset_path,
        train=False,
        preprocessing=None,
        pairs_per_scene=25,
    )
    dataset.build_dataset()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=False)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i % 10 != 0:
            continue

        box1, valid_uv1, box2, valid_uv2, bbox_valid = numpy_overlap_box(
            batch['intrinsics1'][0].numpy(),
            batch['depth1'][0].numpy(),
            batch['pose1'][0].numpy(),
            batch['bbox1'][0].numpy(),
            batch['ratio1'][0].numpy(),
            batch['intrinsics2'][0].numpy(),
            batch['depth2'][0].numpy(),
            batch['pose2'][0].numpy(),
            batch['bbox2'][0].numpy(),
            batch['ratio2'][0].numpy(),
        )
        if bbox_valid:
            # Visualization groundtruth
            visualize_box(
                batch['image0'][0] * 255,
                box1,
                valid_uv1,
                batch['depth1'][0],
                batch['image1'][0] * 255,
                box2,
                valid_uv2,
                batch['depth2'][0],
                batch['file_name'][0],
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate megadepth image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--scene_list_path',
        type=str,
        default='assets/megadepth_validation.txt',
        help='Path to the list of scenes',
    )
    parser.add_argument(
        '--scene_info_path',
        type=str,
        default='assets/megadepth/',
        help='Path to the list of image pairs',
    )
    parser.add_argument('--dataset_path',
                        type=str,
                        default='',
                        help='path to the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='num_workers')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='node rank for distributed training')
    args = parser.parse_args()

    main(**args.__dict__)
