#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File    :   build_model.py
@Time    :   2021/09/28 12:05:27
@Author  :   AbyssGaze 
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import argparse
import math
import os
from collections import defaultdict
from pathlib import Path
from pprint import pformat

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from skimage import measure, transform
from tqdm import tqdm

from .core import extract_features, extractors, match_features, matchers
from .core.utils.base_model import dynamic_load
from .core.utils.utils import (
    make_matching_plot,
    mask_filter,
    read_image,
    read_mask_image,
    read_outer_rectangle,
)

torch.set_grad_enabled(False)


class Matching(torch.nn.Module):
    def __init__(self, config={}, model_path=Path("weights/")):
        super(Matching, self).__init__()
        self.config = config
        if not self.config["direct"]:
            self.extractor = dynamic_load(
                extractors, config["extractor"]["model"]["name"]
            )(config["extractor"]["model"], model_path)
        self.matcher = dynamic_load(matchers, config["matcher"]["model"]["name"])(
            config["matcher"]["model"], model_path
        )

    def forward(self, data):
        """Run extractors and matchers
        Extractor is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        if self.config["direct"]:
            return self.matcher(data)
        pred = {}
        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if "keypoints0" not in data:
            pred0 = self.extractor({"image": data["image0"]})
            pred.update(dict((k + "0", v) for k, v in pred0.items()))
        if "keypoints1" not in data:
            pred1 = self.extractor({"image": data["image1"]})
            pred.update(dict((k + "1", v) for k, v in pred1.items()))
        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data.update(pred)
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        matches = self.matcher(data)
        pred.update(matches)
        return pred


class MatchingTemplate(torch.nn.Module):
    def __init__(self, config={}, model_path=Path("weights/"), device="cuda"):
        super(MatchingTemplate, self).__init__()
        self.config = config
        if not self.config["direct"]:
            self.extractor = dynamic_load(
                extractors, config["extractor"]["model"]["name"]
            )(config["extractor"]["model"], model_path, device).to(device)
        self.matcher = dynamic_load(matchers, config["matcher"]["model"]["name"])(
            config["matcher"]["model"], model_path, device
        ).to(device)

    @torch.no_grad()
    def forward_extractor(self, data):
        pred = {}
        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if "mask" in data.keys():
            pred0 = self.extractor({"image": data["image"], "mask": data["mask"]})
        else:
            pred0 = self.extractor({"image": data["image"]})
        pred.update(dict((k, v) for k, v in pred0.items()))
        return pred

    @torch.no_grad()
    def forward(self, data):
        """Run extractors and matchers
        Extractor is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        if self.config["direct"]:
            return self.matcher(data)
        pred = {}
        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if "keypoints1" not in data:
            pred1 = self.extractor({"image": data["image1"]})
            pred.update(dict((k + "1", v) for k, v in pred1.items()))
        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data.update(pred)
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        matches = self.matcher(data)
        pred.update({"keypoints0": data["keypoints0"], "scores0": data["scores0"]})
        pred.update(matches)
        return pred


class Extracting(torch.nn.Module):
    """Image Matching Frontend (extractor + matcher)"""

    def __init__(self, config={}, model_path=Path("weights/")):
        super(Extracting, self).__init__()
        self.config = config
        self.extractor = dynamic_load(extractors, config["extractor"]["model"]["name"])(
            config["extractor"]["model"], model_path
        )

    def forward(self, data):
        pred = {}
        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if "mask" in data.keys():
            pred0 = self.extractor({"image": data["image"], "mask": data["mask"]})
        else:
            pred0 = self.extractor({"image": data["image"]})
        pred.update(dict((k, v) for k, v in pred0.items()))
        return pred


def build_model(
    extractor,
    matcher,
    device,
    model_path="",
    landmark=False,
    direct=False,
    template=False,
):
    """Building extractor and matcher model

    Args:
        extractor (str): keypoints extractor methods in ['superpoint_aachen', 'superpoint_inloc',
                        'd2net-ss', 'r2d2-desc','context-desc', 'landmark', 'aslfeat-desc']
        matcher (str): keypoints matche methods in ['superglue_outdoor',
                        'superglue_indoor', 'NN', 'disk', 'cotr', 'loftr']
        model_path (str, optional): extractor and matcher weights folder. Defaults to ''.
        landmark (bool, optional): Keypoints extraction with landmarks. Defaults to False.
        direct (bool, optional): Match images directe without keypoints extraction. Defaults to False.

    Returns:
        model: extractor and matcher model
        config: extractor and matcher config
    """

    extractor_conf = extract_features.confs[extractor]
    matcher_conf = match_features.confs[matcher]
    config = {
        "landmark": landmark,
        "extractor": extractor_conf,
        "matcher": matcher_conf,
        "direct": direct,
    }
    # Load the SuperPoint and SuperGlue models.
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if template:
        model = MatchingTemplate(config, Path(model_path), device).eval()
    else:
        model = Matching(config, Path(model_path)).eval().to(device)
    return model, config


def get_matches(
    name0, name1, model, config, resize=[-1], with_desc=False, landmarks=None
):
    """Input image pair and output matches

    Args:
        name0 (str): first image path
        name1 (str): second image path
        model : extractor and matcher model
        config (dict): extractor and matcher config
        resize (list, optional): parameters of resize. Defaults to [-1].
        with_desc (bool, optional): return without descriptors. Defaults to False.
        landmarks (np.array, optional): landmarks of keypoints(same as template keypoints). Defaults to None.

    Returns:
        dict: return keypoints(with descriptor) and matches with confidence
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gray = config["extractor"]["preprocessing"]["grayscale"]
    # Load the image pair.
    align = ""
    if "disk" in config["extractor"]["output"]:
        align = "disk"
    elif "loftr" in config["extractor"]["output"]:
        align = "loftr"

    image0, inp0, scales0 = read_image(name0, device, resize, 0, True, gray, align)
    image1, inp1, scales1 = read_image(name1, device, resize, 0, True, gray, align)
    if image0 is None or image1 is None:
        print("Problem reading image pair: {} {}".format(input / name0, input / name1))
        exit(1)

    # Perform the matching.
    if landmarks:
        template_kpts = landmarks / scales0
        pred = model({"image0": inp0, "image1": inp1, "landmark": template_kpts})
    else:
        pred = model({"image0": inp0, "image1": inp1})

    pred = dict((k, v[0].cpu().numpy()) for k, v in pred.items())
    output = {}
    kpts0, kpts1 = pred["keypoints0"] * scales0, pred["keypoints1"] * scales1
    matches, conf = pred["matches0"], pred["matching_scores0"]
    if with_desc:
        desc0, desc1 = pred["descriptors0"], pred["descriptors1"]
        return {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "matches": matches,
            "mconf": conf,
            "descriptors0": desc0,
            "descriptors1": desc1,
        }
    return {
        "keypoints0": kpts0,
        "keypoints1": kpts1,
        "matches": matches,
        "mconf": conf,
    }


def get_pose(name0, name1, model, config, resize=[-1], landmarks=None, mode="H"):
    """Direct calculation of image pair relative pose

    Args:
        name0 (str): first image path
        name1 (str): second image path
        model : extractor and matcher model
        config (dict): extractor and matcher config
        resize (list, optional): parameters of resize. Defaults to [-1].
        landmarks (np.array, optional): landmarks of keypoints(same as template keypoints). Defaults to None.
        mode (str, optional): affine matrix of homography matrix. Defaults to 'H'.

    Raises:
        ValueError: mode not supported

    Returns:
        dict: return pose and matches points
    """
    output = get_matches(name0, name1, model, config, resize, False, landmarks)
    valid = output["matches"] > -1
    mkpts0 = output["keypoints0"][valid]
    mkpts1 = output["keypoints1"][output["matches"][valid]]
    if mode == "H":
        M, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    elif mode == "A":
        M, inliers = cv2.getAffineTransform(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    elif mode == "S":
        model, inliers = measure.ransac(
            (mkpts1, mkpts0),
            transform.SimilarityTransform,
            min_samples=2,
            residual_threshold=5,
        )
        M = model.params
    else:
        raise ValueError(f"Pose type {mode} not supported.")
    mkpts0 = mkpts0[inliers.ravel() == 1]
    mkpts1 = mkpts1[inliers.ravel() == 1]
    return {
        "pose": M,
        "mkpts0": mkpts0,
        "mkpts1": mkpts1,
    }


def post_process(template_img, cur_img, M, mask):
    cur_img = cv2.warpPerspective(
        cur_img, M, (cur_img.shape[1], cur_img.shape[0]), borderValue=0.0
    )
    std_roi = template_img * mask
    cur_roi = cur_img * mask

    binary_thresh = (np.sum(std_roi) // np.sum(mask)) // 3

    ret, roi_binary = cv2.threshold(std_roi, binary_thresh, 1, cv2.THRESH_BINARY)
    ret, cur_binary = cv2.threshold(cur_roi, binary_thresh, 1, cv2.THRESH_BINARY)
    binary_diff = math.fabs(int(np.sum(roi_binary)) - int(np.sum(cur_binary)))
    similar_coeff = binary_diff / np.sum(mask)
    return similar_coeff, cur_img


class JabilPipeline:
    def __init__(
        self,
        template_path="assets/template",
        extractor_mode="disk-desc",
        matcher_mode="superglue_jabil",
        weight_path="weights",
        device="cuda:0",
        resize=[612],
        roi_offset=200,
        similar_thresh=0.7,
    ):
        self.matcher, self.config = build_model(
            extractor_mode,
            matcher_mode,
            weight_path,
            landmark=False,
            direct=False,
            template=True,
        )
        self.extractor = Extracting(self.config, weight_path).eval().to(device)
        self.template_path = template_path
        self.device = device
        self.resize = resize
        self.roi_offset = roi_offset
        self.gray = self.config["extractor"]["preprocessing"]["grayscale"]
        self.align = ""
        if "disk" in self.config["extractor"]["output"]:
            self.align = "disk"
        elif "loftr" in self.config["matcher"]["output"]:
            self.align = "loftr"
        self.template_info = self._template_extract()
        self.similar_thresh = similar_thresh

    def _mask_filter(self, img):
        _, mask = cv2.threshold(img.astype("uint8"), 30, 255, cv2.THRESH_BINARY)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
        mask = cv2.dilate(mask, verticalStructure)
        return mask

    def _template_extract(self):
        template_files = os.listdir(self.template_path)
        template_infos = defaultdict(dict)
        for f in template_files:
            pos = f.split("/")[-1][:-5]
            if int(pos) in [
                2,
                4,
                5,
                6,
                8,
                10,
                12,
                13,
                14,
                16,
                18,
                20,
                32,
                34,
                36,
                48,
                50,
                52,
                54,
                55,
                56,
                58,
                59,
                60,
                62,
                63,
                64,
                66,
                68,
                80,
                82,
                84,
            ]:
                continue
            _, roi, _ = read_outer_rectangle(
                f.replace("jpg", "json"), offset=self.roi_offset
            )
            img, inp, scale = read_mask_image(
                f, self.device, self.resize, 0, True, self.gray, self.align, rect=roi
            )

            if img is None:
                print("Problem reading image: {}/{}".format(input, f))
                exit(1)

            mask = self.mask_filter(img)
            pred = self.extractor({"image": inp})
            kpts = pred["keypoints"][0].cpu().numpy()
            desc = pred["descriptors"][0].cpu().numpy()
            score = pred["scores"][0].cpu().numpy()
            template_infos[pos]["keypoint"] = kpts
            template_infos[pos]["descriptor"] = desc
            template_infos[pos]["score"] = score
            template_infos[pos]["scale"] = scale
            template_infos[pos]["mask"] = mask
            template_infos[pos]["image"] = img
            template_infos[pos]["roi"] = roi
        return template_infos

    def pose_estimation(self, pred):
        kpts0 = pred[
            "keypoints0"
        ]  # *template_scales[pos]# + np.array([template_rois[pos][0], template_rois[pos][1]])
        kpts1 = pred[
            "keypoints1"
        ]  # *scales1# + np.array([template_rois[pos][0], template_rois[pos][1]])
        matches, conf = pred["matches0"], pred["matching_scores0"]
        valid = matches > -1
        mconf = conf[valid]
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        if mkpts0.shape[0] < 4:
            return np.eye(3), 0
        model, inliers = measure.ransac(
            (mkpts1, mkpts0),
            transform.SimilarityTransform,
            min_samples=2,
            residual_threshold=5,
        )
        M = model.params
        return M, 1

    def _similarity_evaluation(self, template_img, template_mask, cur_img, M):
        if (M == np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])).all():
            return 1.0

        cur_img = cv2.warpPerspective(
            cur_img, M, (cur_img.shape[1], cur_img.shape[0]), borderValue=0.0
        )
        template_roi = template_img * template_mask
        cur_roi = cur_img * template_mask

        binary_thresh = (np.sum(template_roi) // np.sum(template_mask)) // 3
        ret, roi_binary = cv2.threshold(
            template_roi, binary_thresh, 1, cv2.THRESH_BINARY
        )
        ret, cur_binary = cv2.threshold(cur_roi, binary_thresh, 1, cv2.THRESH_BINARY)
        binary_diff = math.fabs(int(np.sum(roi_binary)) - int(np.sum(cur_binary)))
        similar_coeff = binary_diff / np.sum(template_mask)
        return similar_coeff

    def process(self, img_file):
        pos = img_file.split("/")[-1].split("_")[2][1:]
        img, inp, scales = read_mask_image(
            img_file,
            self.device,
            self.resize,
            0,
            True,
            self.gray,
            self.align,
            rect=self.template_info[pos]["roi"],
        )

        pred = self.matcher(
            {
                "keypoints0": self.template_info[pos]["keypoint"],
                "scores0": self.template_info[pos]["score"],
                "descriptors0": self.template_info[pos]["descriptor"],
                "image0": self.template_info[pos]["image"],
                "image1": inp,
            }
        )
        pred = dict((k, v[0].cpu().numpy()) for k, v in pred.items())
        M, ret = self.pose_estimation(pred)
        if not ret:
            return M, ret, "Pose estimation failed with less matches."
        similar_coeff = self._similarity_evaluation(
            self.template_info[pos]["image"], self.template_info[pos]["mask"], img, M
        )
        if similar_coeff < self.similar_thresh:
            reason = "Similarity: {} > {}".format(similar_coeff, self.similar_thresh)
            ret = 0
        return M, ret, reason
