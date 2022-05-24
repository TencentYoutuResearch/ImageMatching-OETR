#!/usr/bin/env python
"""
@File    :   utils.py
@Time    :   2021/06/17 18:06:02
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import logging
import math

import cv2
import numpy as np
from scipy.spatial import distance


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, 'a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def normalize_keypoints(keypoints, K):
    """Normalize keypoints using the calibration data."""

    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

    return keypoints


def unnormalize_keypoints(keypoints, K):
    """Undo the normalization of the keypoints using the calibration data."""
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = keypoints * np.array([[f_x, f_y]]) + np.array([[C_x, C_y]])

    return keypoints


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    """

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython

        IPython.embed()

    return err_q, err_t


def eval_essential_matrix(p1n, p2n, E, dR, dt):
    if len(p1n) != len(p2n):
        raise RuntimeError('Size mismatch in the keypoint lists')

    if p1n.shape[0] < 5:
        return np.pi, np.pi / 2

    if E.size > 0:
        _, R, t, _ = cv2.recoverPose(E, p1n, p2n)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except ValueError:
            err_q = np.pi
            err_t = np.pi / 2

    else:
        err_q = np.pi
        err_t = np.pi / 2

    return err_q, err_t


def get_projected_kp(x1, x2, d1, d2, dR, dT):
    # Append depth to key points
    y1 = np.concatenate([x1 * d1, d1], axis=1)
    y2 = np.concatenate([x2 * d2, d2], axis=1)

    # Project points from one image to another image
    y1p = np.matmul(dR[None], y1[..., None]) + dT[None]
    y2p = (np.matmul(np.transpose(dR)[None], y2[..., None]) -
           np.matmul(np.transpose(dR), dT)[None])

    # Move back to canonical plane
    x1p = np.squeeze(y1p[:, 0:2] / y1p[:, [2]])
    x2p = np.squeeze(y2p[:, 0:2] / y2p[:, [2]])

    return x1p, x2p


def get_repeatability(kp1n_p, kp2n, th_list):
    if kp1n_p.shape[0] == 0:
        return [0] * len(th_list)

    # Construct distance matrix
    # dis_mat = (np.tile(np.dot(kp1n_p * kp1n_p, np.ones([2, 1])),
    #                    (1, kp2n.shape[0])) +
    #            np.tile(np.transpose(np.dot(kp2n * kp2n, np.ones([2, 1]))),
    #                    (kp1n_p.shape[0], 1)) -
    #            2 * np.dot(kp1n_p, np.transpose(kp2n)))

    # Eduard: Extremely slow, this should be better
    dis_mat = distance.cdist(kp1n_p, kp2n, metric='sqeuclidean')

    # Get min for each point in kp1n_p
    min_array = np.amin(dis_mat, 1)

    # Calculate repeatability
    rep_score_list = []
    for th in th_list:
        rep_score_list.append((min_array < th * th).sum() / kp1n_p.shape[0])

    return rep_score_list


def get_matching_score(kpts1, kpts2):
    pass


def np_skew_symmetric(v):

    zero = np.zeros_like(v[:, 0])

    M = np.stack(
        [
            zero,
            -v[:, 2],
            v[:, 1],
            v[:, 2],
            zero,
            -v[:, 0],
            -v[:, 1],
            v[:, 0],
            zero,
        ],
        axis=1,
    )

    return M


def get_episym(x1n, x2n, dR, dt):

    # Fix crash when passing a single match
    if x1n.ndim == 1:
        x1n = x1n[None, ...]
        x2n = x2n[None, ...]

    num_pts = len(x1n)

    # Make homogeneous coordinates
    x1n = np.concatenate([x1n, np.ones((num_pts, 1))], axis=-1).reshape(
        (-1, 3, 1))
    x2n = np.concatenate([x2n, np.ones((num_pts, 1))], axis=-1).reshape(
        (-1, 3, 1))

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(
        np.matmul(np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
                  dR).reshape(-1, 3, 3),
        num_pts,
        axis=0,
    )

    x2Fx1 = np.matmul(x2n.transpose(0, 2, 1), np.matmul(F, x1n)).flatten()
    Fx1 = np.matmul(F, x1n).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2n).reshape(-1, 3)

    ys = x2Fx1**2 * (1.0 / (Fx1[..., 0]**2 + Fx1[..., 1]**2) + 1.0 /
                     (Ftx2[..., 0]**2 + Ftx2[..., 1]**2))

    return ys.flatten()


def get_truesym(x1, x2, x1p, x2p):
    if len(x1) == 0 or len(x1p) == 0:
        return []

    ys1 = np.sqrt(np.sum((x1p - x2) * (x1p - x2), axis=1))
    ys2 = np.sqrt(np.sum((x2p - x1) * (x2p - x1), axis=1))
    ys = (ys1 + ys2) / 2
    ys = ys2

    return ys.flatten()


def eval_match_score(kp1, kp2, kp1n, kp2n, kp1p, kp2p, d1, d2, inl, dR, dT):
    # Fail silently if there are no inliers
    if inl.size == 0:
        return np.array([]), np.array([])

    # Fix crash when this is a single element
    # Should not happen but it seems that it does?
    if inl.ndim == 1:
        inl = inl[..., None]

    kp1_inl = kp1[inl[0]]
    kp2_inl = kp2[inl[1]]
    kp1p_inl = kp1p[inl[0]]
    kp2p_inl = kp2p[inl[1]]
    kp1n_inl = kp1n[inl[0]]
    kp2n_inl = kp2n[inl[1]]
    d1_inl = d1[inl[0]]
    d2_inl = d2[inl[1]]

    nonzero_index = np.nonzero(np.squeeze(d1_inl * d2_inl))

    # Get the geodesic distance in normalized coordinates
    geod_d = get_episym(kp1n_inl, kp2n_inl, dR, dT)

    # Get the projected distance in image coordinates
    true_d = get_truesym(
        kp1_inl[nonzero_index],
        kp2_inl[nonzero_index],
        kp1p_inl[nonzero_index],
        kp2p_inl[nonzero_index],
    )

    return geod_d, true_d


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


def pose_acc(errors, thresholds):
    acc_ratio = []
    for t in thresholds:
        acc = (np.array(errors) < t).sum() / len(errors)
        acc_ratio.append(acc)
    return acc_ratio


def pose_mAA(errors):
    bars = np.arange(11)
    qt_hist, _ = np.histogram(errors, bars)
    num_pair = float(len(errors))
    qt_hist = qt_hist.astype(float) / num_pair
    qt_acc = np.cumsum(qt_hist)
    return np.mean(qt_acc)
