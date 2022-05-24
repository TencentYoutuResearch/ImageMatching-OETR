#!/usr/bin/env python
"""
@File    :   icp.py
@Time    :   2021/06/25 17:04:05
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import cv2
import numpy as np

from ..utils.base_model import BaseModel


class ICPRegister(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_

    Returns:
        _type_: _description_
    """

    default_conf = {
        'max_points': 10000,
        'max_iter': 100,
        'ransac_threshold': 50,
        'intensity_thresh': 50,
    }
    required_inputs = ['image0', 'image1', 'mask0', 'mask1']

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}
        self.max_points = self.conf['max_points']
        self.max_iter = self.conf['max_iter']
        self.ransac_threshold = self.conf['ransac_threshold']
        self.intensity_thresh = self.conf['intensity_thresh']

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def del_miss(self, matches, dist, max_dist, th_rate=0.8):
        th_dist = max_dist * th_rate
        mask = dist.T[0] < th_dist
        return matches[:, mask]

    def get_contours(self, image, mask=None):
        h, w = image.shape[:2]
        if mask is None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, self.intensity_thresh, 255, 0)
        else:
            thresh = mask

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)

        areas = [cv2.contourArea(c) for c in contours]
        if areas:
            contours = np.array(self.flatten(contours)).squeeze(1)  # N, 2
            valid = ((contours[:, 0] > 5)
                     & (contours[:, 0] < w - 5)
                     & (contours[:, 1] > 5)
                     & (contours[:, 1] < h - 5))
            return contours[valid]
        else:
            return []

    def icp(self, d1, d2):
        src = np.array([d1.T], copy=True).astype(np.float32)
        dst = np.array([d2.T], copy=True).astype(np.float32)
        knn = cv2.ml.KNearest_create()
        knn.setAlgorithmType(2)
        responses = np.array(range(len(d1[0]))).astype(np.float32)
        knn.train(src[0], cv2.ml.ROW_SAMPLE, responses)

        Tr = np.array([[np.cos(0), -np.sin(0), 0], [np.sin(0),
                                                    np.cos(0), 0], [0, 0, 1]])

        dst = cv2.transform(dst, Tr[0:2])
        max_dist = 9999999.0

        last_error = max_dist

        for _ in range(self.max_iter):

            _, results, _, dist = knn.findNearest(dst[0], 1)
            dist = np.linalg.norm(dst[0] - src[0][results[:, 0]], axis=1)
            indices = results.astype(np.int32).T
            matches = np.stack([np.arange(len(dst[0])), indices[0]], axis=0)
            matches = self.del_miss(matches, dist, max_dist)

            if matches.size == 0:
                return None
            T, _ = cv2.estimateAffinePartial2D(
                dst[0, matches[0]],
                src[0, matches[1]],
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
            )

            max_dist = np.max(dist)
            dst = cv2.transform(dst, T)
            Tr = np.dot(np.vstack((T, [0, 0, 1])), Tr)
            error = dist[dist < self.ransac_threshold].mean()
            if last_error - error < 1e-3:
                break
            last_error = error

        return Tr

    def _forward(self, data):
        c1 = self.get_contours(data['image0'], data['mask0'])
        c2 = self.get_contours(data['image1'], data['mask1'])
        if len(c1) > 0 and len(c2) > 0:
            c1 = c1[np.random.choice(len(c1), self.max_points)].T
            c2 = c2[np.random.choice(len(c2), self.max_points)].T
            T_0_1 = self.icp(c1, c2)
        else:
            T_0_1 = np.eye(3)
        return {'T_0_1': T_0_1}
