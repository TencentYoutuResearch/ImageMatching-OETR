#!/usr/bin/env python
"""
@File    :   build_model.py
@Time    :   2021/09/28 12:05:27
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
from pathlib import Path

import cv2
import torch

from .core import extract_features, extractors, match_features, matchers
from .core.utils.base_model import dynamic_load
from .core.utils.utils import read_image

torch.set_grad_enabled(False)


class Matching(torch.nn.Module):
    def __init__(self, config=None, model_path=Path('weights/')):
        super(Matching, self).__init__()
        # if config is None:
        #     self.config = {}
        # else:
        self.config = config
        if not self.config['direct']:
            self.extractor = dynamic_load(
                extractors, config['extractor']['model']['name'])(
                    config['extractor']['model'], model_path)
        self.matcher = dynamic_load(matchers,
                                    config['matcher']['model']['name'])(
                                        config['matcher']['model'], model_path)

    def forward(self, data):
        """Run extractors and matchers
        Extractor is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        if self.config['direct']:
            return self.matcher(data)
        pred = extract_process(self.extractor, data)
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


def extract_process(extractor, data):
    pred = {}
    # Extract SuperPoint (keypoints, scores, descriptors) if not provided
    if 'keypoints0' not in data:
        pred0 = extractor({'image': data['image0']})
        pred.update(dict((k + '0', v) for k, v in pred0.items()))
    if 'keypoints1' not in data:
        pred1 = extractor({'image': data['image1']})
        pred.update(dict((k + '1', v) for k, v in pred1.items()))
    return pred


def build_model(extractor,
                matcher,
                model_path='',
                landmark=False,
                direct=False):
    """Building extractor and matcher model.

    Args:
        extractor (str): keypoints extractor methods in ['superpoint_aachen',
            'superpoint_inloc', 'd2net-ss', 'r2d2-desc','context-desc',
            'landmark', 'aslfeat-desc']
        matcher (str): keypoints matche methods in ['superglue_outdoor',
            'superglue_indoor', 'NN', 'disk', 'cotr', 'loftr']
        model_path (str, optional): extractor and matcher weights folder.
        landmark (bool, optional): Keypoints extraction with landmarks.
        direct (bool, optional): Match images without keypoints extraction.

    Returns:
        model: extractor and matcher model
        config: extractor and matcher config
    """

    extractor_conf = extract_features.confs[extractor]
    matcher_conf = match_features.confs[matcher]
    config = {
        'landmark': landmark,
        'extractor': extractor_conf,
        'matcher': matcher_conf,
        'direct': direct,
    }
    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Matching(config, Path(model_path)).eval().to(device)
    return model, config


def get_matches(name0,
                name1,
                model,
                config,
                resize=None,
                with_desc=False,
                landmarks=None):
    """Input image pair and output matches.

    Args:
        name0 (str): first image path
        name1 (str): second image path
        model : extractor and matcher model
        config (dict): extractor and matcher config
        resize (list, optional): parameters of resize. Defaults to [-1].
        with_desc (bool, optional): return without descriptors.
        landmarks (np.array, optional):
            landmarks of keypoints(same as template keypoints).

    Returns:
        dict: return keypoints(with descriptor) and matches with confidence
    """
    if resize is None:
        resize = [-1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gray = config['extractor']['preprocessing']['grayscale']
    # Load the image pair.
    align = ''
    if 'disk' in config['extractor']['output']:
        align = 'disk'
    elif 'loftr' in config['extractor']['output']:
        align = 'loftr'

    image0, inp0, scales0 = read_image(name0, device, resize, 0, True, gray,
                                       align)
    image1, inp1, scales1 = read_image(name1, device, resize, 0, True, gray,
                                       align)
    if image0 is None or image1 is None:
        print('Problem reading image pair: {} {}'.format(
            input / name0, input / name1))
        exit(1)

    # Perform the matching.
    if landmarks:
        template_kpts = landmarks / scales0
        pred = model({
            'image0': inp0,
            'image1': inp1,
            'landmark': template_kpts
        })
    else:
        pred = model({'image0': inp0, 'image1': inp1})

    pred = dict((k, v[0].cpu().numpy()) for k, v in pred.items())
    kpts0, kpts1 = pred['keypoints0'] * scales0, pred['keypoints1'] * scales1
    matches, conf = pred['matches0'], pred['matching_scores0']
    if with_desc:
        desc0, desc1 = pred['descriptors0'], pred['descriptors1']
        return {
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'matches': matches,
            'mconf': conf,
            'descriptors0': desc0,
            'descriptors1': desc1,
        }
    return {
        'keypoints0': kpts0,
        'keypoints1': kpts1,
        'matches': matches,
        'mconf': conf,
    }


def get_pose(name0,
             name1,
             model,
             config,
             resize=None,
             landmarks=None,
             mode='H'):
    """Direct calculation of image pair relative pose.

    Args:
        name0 (str): first image path
        name1 (str): second image path
        model : extractor and matcher model
        config (dict): extractor and matcher config
        resize (list, optional): parameters of resize. Defaults to [-1].
        landmarks (np.array, optional):
            landmarks of keypoints(same as template keypoints).
        mode (str, optional): affine matrix of homography matrix.

    Raises:
        ValueError: mode not supported

    Returns:
        dict: return pose and matches points
    """
    if resize is None:
        resize = [-1]
    output = get_matches(name0, name1, model, config, resize, False, landmarks)
    valid = output['matches'] > -1
    mkpts0 = output['keypoints0'][valid]
    mkpts1 = output['keypoints1'][output['matches'][valid]]
    if mode == 'H':
        M, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    elif mode == 'A':
        M, inliers = cv2.getAffineTransform(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    else:
        raise ValueError(f'Pose type {mode} not supported.')
    mkpts0 = mkpts0[inliers.ravel() == 1]
    mkpts1 = mkpts1[inliers.ravel() == 1]
    return {
        'pose': M,
        'mkpts0': mkpts0,
        'mkpts1': mkpts1,
    }
