#!/usr/bin/env python
"""
@File    :   evaluation.py
@Time    :   2021/06/17 17:59:14
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import cv2
import numpy as np
from skimage import measure, transform

from .utils import (eval_essential_matrix, eval_match_score, get_projected_kp,
                    get_repeatability, normalize_keypoints,
                    unnormalize_keypoints)


def h_evaluate(H, kpts0, kpts1, matches):
    pos_a = kpts0[matches[:, 0], :2]
    pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
    pos_b_proj_h = np.transpose(np.dot(H, np.transpose(pos_a_h)))
    pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, 2:]

    pos_b = kpts1[matches[:, 1], :2]

    dist = np.sqrt(np.sum((pos_b - pos_b_proj).numpy()**2, axis=1))
    return dist


def homo_trans(coord, H):
    kpt_num = coord.shape[0]
    homo_coord = np.concatenate((coord, np.ones((kpt_num, 1))), axis=-1)
    proj_coord = np.matmul(H, homo_coord.T).T
    proj_coord = proj_coord / proj_coord[:, 2][..., None]
    proj_coord = proj_coord[:, 0:2]
    return proj_coord


def pr_evaluate(kpts0,
                kpts1,
                matches,
                template_kpts,
                query_kpts,
                use_opencv=False):
    ref_coord = kpts0[matches[:, 0], :2]
    test_coord = kpts1[matches[:, 1], :2]
    if test_coord.shape[0] < 4:
        return query_kpts.numpy(), template_kpts.numpy()

    if use_opencv:
        pred_homo, _ = cv2.estimateAffinePartial2D(
            ref_coord.numpy(),
            test_coord.numpy(),
            method=cv2.RANSAC,
            ransacReprojThreshold=20,
            maxIters=3000,
        )
        pred_homo = np.vstack((pred_homo, [0, 0, 1]))
    else:
        model, _ = measure.ransac(
            (ref_coord.numpy(), test_coord.numpy()),
            transform.SimilarityTransform,
            min_samples=2,
            residual_threshold=20,
        )
        pred_homo = model.params
    if pred_homo is None:
        pred_homo = np.eye(3)
    warped_template_corners = homo_trans(template_kpts, pred_homo)
    # pred_homo, _ = cv2.estimateAffine2D(
    #     test_coord.numpy(), ref_coord.numpy(), cv2.RANSAC, 1, maxIters=3000)
    warped_query_corners = homo_trans(query_kpts, np.linalg.inv(pred_homo))

    return warped_template_corners, warped_query_corners


def pr_evaluate_directly(pred_homo, template_kpts, query_kpts):
    warped_template_corners = homo_trans(template_kpts, pred_homo)
    warped_query_corners = homo_trans(query_kpts, np.linalg.inv(pred_homo))

    return warped_template_corners, warped_query_corners


def pr_evaluate_cv(kpts0, kpts1, matches, template_kpts, query_kpts):
    ref_coord = kpts0[matches[:, 0], :2]
    test_coord = kpts1[matches[:, 1], :2]
    if test_coord.shape[0] < 4:
        return query_kpts.numpy(), template_kpts.numpy()
    pred_homo, _ = cv2.estimateAffine2D(ref_coord.numpy(), test_coord.numpy(),
                                        cv2.RANSAC)
    pred_homo = np.vstack((pred_homo, np.array([0, 0, 1])))

    warped_template_corners = homo_trans(template_kpts, pred_homo)
    warped_query_corners = homo_trans(query_kpts, np.linalg.inv(pred_homo))

    return warped_template_corners, warped_query_corners


def pose_evaluate(
    depth0,
    depth1,
    kpts0,
    kpts1,
    K1,
    K2,
    dR,
    dT,
    E,
    inl_prematch,
    inl_refined,
    inl_geom,
    thresh=3,
):
    """Computes the stereo metrics."""

    # Compute error in R, T
    kpts0n = normalize_keypoints(kpts0, K1)
    kpts1n = normalize_keypoints(kpts1, K2)
    err_q, err_t = eval_essential_matrix(kpts0n[inl_geom[0]],
                                         kpts1n[inl_geom[1]], E, dR, dT)

    # If the dataset does not contain depth information, there is nothing else
    # to do.
    if depth0 is None:
        return [], [], err_q, err_t, [], True

    # Clip keypoints based on shape of matches
    kpts0 = kpts0[:, :2]
    kpts1 = kpts1[:, :2]

    # depth map
    img0_shp = depth0.shape
    img1_shp = depth1.shape

    # Get depth for each keypoint
    kpts0_int = np.round(kpts0).astype(int)
    kpts1_int = np.round(kpts1).astype(int)

    # Some methods can give out-of-bounds keypoints close to image boundaries
    # Safely marked them as occluded
    valid1 = ((kpts0_int[:, 0] >= 0)
              & (kpts0_int[:, 0] < depth0.shape[1])
              & (kpts0_int[:, 1] >= 0)
              & (kpts0_int[:, 1] < depth0.shape[0]))
    valid2 = ((kpts1_int[:, 0] >= 0)
              & (kpts1_int[:, 0] < depth1.shape[1])
              & (kpts1_int[:, 1] >= 0)
              & (kpts1_int[:, 1] < depth1.shape[0]))
    d1 = np.zeros((kpts0_int.shape[0], 1))
    d2 = np.zeros((kpts1_int.shape[0], 1))
    d1[valid1, 0] = depth0[kpts0_int[valid1, 1], kpts0_int[valid1, 0]]
    d2[valid2, 0] = depth1[kpts1_int[valid2, 1], kpts1_int[valid2, 0]]

    # Project the keypoints using depth
    kpts0n_p, kpts1n_p = get_projected_kp(kpts0n, kpts1n, d1, d2, dR, dT)
    kpts0_p = unnormalize_keypoints(kpts0n_p, K2)
    kpts1_p = unnormalize_keypoints(kpts1n_p, K1)

    # Get non zero depth index
    d1_nonzero_idx = np.nonzero(np.squeeze(d1))
    d2_nonzero_idx = np.nonzero(np.squeeze(d2))

    # Get index of projected kp inside image
    kpts0_p_valid_idx = np.where((kpts0_p[:, 0] < img1_shp[1])
                                 & (kpts0_p[:, 1] < img1_shp[0]))
    kpts1_p_valid_idx = np.where((kpts1_p[:, 0] < img0_shp[1])
                                 & (kpts1_p[:, 1] < img0_shp[0]))
    # print('Preprocessing: {}'.format(time() - t))

    # Calculate repeatability
    # Thresholds are hardcoded
    rep_s_list_1 = get_repeatability(
        kpts0_p[np.intersect1d(kpts0_p_valid_idx, d1_nonzero_idx)], kpts1,
        thresh)
    rep_s_list_2 = get_repeatability(
        kpts1_p[np.intersect1d(kpts1_p_valid_idx, d2_nonzero_idx)], kpts0,
        thresh)
    rep_s_list = [(rep_s_0 + rep_s_1) / 2
                  for rep_s_0, rep_s_1 in zip(rep_s_list_1, rep_s_list_2)]

    # Evaluate matching score after initial matching
    geod_d_list = []
    true_d_list = []
    geod_d, true_d = eval_match_score(kpts0, kpts1, kpts0n, kpts1n, kpts0_p,
                                      kpts1_p, d1, d2, inl_prematch, dR, dT)
    geod_d_list.append(geod_d)
    true_d_list.append(true_d)

    # Evaluate matching score after inlier refinement
    if inl_refined is None:
        geod_d_list.append([])
        true_d_list.append([])
    else:
        geod_d, true_d = eval_match_score(kpts0, kpts1, kpts0n, kpts1n,
                                          kpts0_p, kpts1_p, d1, d2,
                                          inl_refined, dR, dT)
        geod_d_list.append(geod_d)
        true_d_list.append(true_d)

    # Evaluate matching score after final geom
    geod_d, true_d = eval_match_score(kpts0, kpts1, kpts0n, kpts1n, kpts0_p,
                                      kpts1_p, d1, d2, inl_geom, dR, dT)
    geod_d_list.append(geod_d)
    true_d_list.append(true_d)

    return geod_d_list, true_d_list, err_q, err_t, rep_s_list, True


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None, []

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(kpts0,
                                   kpts1,
                                   np.eye(3),
                                   threshold=norm_thresh,
                                   prob=conf,
                                   method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E,
                                     kpts0,
                                     kpts1,
                                     np.eye(3),
                                     1e9,
                                     mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret, mask


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([[0, -t2, t1], [t2, 0, -t0], [-t1, t0, 0]])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0] + Ep0[:, 1]) + 1.0 /
                    (Etp1[:, 0] + Etp1[:, 1]))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def validation_error(data, thresh=1.0):
    pt1 = data['kpts0'].float()
    pt2 = data['kpts1'].float()

    batch_error_t = []
    batch_error_R = []
    batch_precision = []
    batch_matching_score = []
    batch_num_correct = []
    batch_epipolar_errors = []
    for i in range(pt1.shape[0]):
        kpts1 = pt1[i]
        kpts2 = pt2[i]
        matches = data['matches'][i]
        mkpts1 = kpts1[matches[0]].cpu().numpy()
        mkpts2 = kpts2[matches[1]].cpu().numpy()
        K1 = data['intrinsics0'][i].cpu().numpy()
        K2 = data['intrinsics1'][i].cpu().numpy()
        T_1to2 = data['pose'][i].view(4, 4).cpu().numpy()

        if 'inparams0' in data.keys() and data['inparams0'] is not None:
            sx1, sy1, tx1, ty1, rx1, ry1 = data['inparams0'][i].cpu().numpy()
            sx2, sy2, tx2, ty2, rx2, ry2 = data['inparams1'][i].cpu().numpy()
            K_n1to1 = np.array(
                [[sx1 / rx1, 0, sx1 * tx1], [0, sy1 / ry1, sy1 * ty1],
                 [0, 0, 1]],
                dtype=np.float,
            )
            K_n2to2 = np.array(
                [[sx2 / rx2, 0, sx2 * tx2], [0, sy2 / ry2, sy2 * ty2],
                 [0, 0, 1]],
                dtype=np.float,
            )
            K1_inv = np.linalg.inv(K1) @ K_n1to1
            K1 = np.linalg.inv(K1_inv)
            K2_inv = np.linalg.inv(K2) @ K_n2to2
            K2 = np.linalg.inv(K2_inv)
            # epi_errs = compute_epipolar_error(mkpts1, mkpts2, T_n2to2@T_1to2@np.linalg.inv(T_n1to1), K1, K2)
            epi_errs = compute_epipolar_error(mkpts1, mkpts2, T_1to2, K1, K2)
        else:
            # K_n1to1 = K_n2to2 = None
            epi_errs = compute_epipolar_error(mkpts1, mkpts2, T_1to2, K1, K2)
        correct = epi_errs < 5e-4
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(kpts1) if len(kpts1) > 0 else 0

        # thresh = 1.  # In pixels relative to resized image size.
        ret, mask = estimate_pose(mkpts1, mkpts2, K1, K2, thresh)
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, _ = ret
            err_t, err_R = compute_pose_error(T_1to2, R, t)

        batch_epipolar_errors.append(epi_errs)
        batch_num_correct.append(num_correct)
        batch_matching_score.append(matching_score)
        batch_precision.append(precision)
        batch_error_R.append(err_R)
        batch_error_t.append(err_t)

        # Write the evaluation results to disk.
        results = {
            'error_t': np.mean(batch_error_t),
            'error_R': np.mean(batch_error_R),
            'precision': np.mean(batch_precision),
            'matching_score': np.mean(batch_matching_score),
            'num_correct': np.mean(batch_num_correct),
            'epipolar_errors': np.mean(batch_epipolar_errors),
            'inliers': mask,
        }
        return results
