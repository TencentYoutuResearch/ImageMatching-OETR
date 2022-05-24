#! /usr/bin/env python3
import argparse
import logging
import pprint
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm


"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
"""
confs = {
    'superpoint_aachen': {
        'output': 'feats-superpoint-n2048-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 2048,
            'keypoint_threshold': 0.005,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    'superpoint_inloc': {
        'output': 'feats-superpoint-n4096-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'd2net-ss': {
        'output': 'feats-d2net-ss',
        'model': {
            'name': 'd2net',
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    'context-desc': {
        'output': 'feats-context-desc',
        'model': {
            'name': 'contextdesc',
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    'r2d2-desc': {
        'output': 'feats-r2d2-desc',
        'model': {
            'name': 'r2d2',
            'reliability_thr': 0.7,
            'repeatability_thr': 0.7,
            'topk': 5000,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'aslfeat-desc': {
        'output': 'feats-aslfeat-desc',
        'model': {
            'name': 'aslfeat',
            'max_keypoints': 2048,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    'disk-desc': {
        'output': 'feats-disk-desc',
        'model': {
            'name': 'disk',
            'topk': 2048,
            'weights': 'disk_depth.pth'
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1280,
        },
    },
    'swin-disk-desc': {
        'output': 'feats-swin-disk-desc',
        'model': {
            'name': 'disk',
            'topk': 2048,
            'weights': 'disk-swin-conv.pth',
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1280,
        },
    },
    'landmark': {
        'output': 'feats-landmark-desc',
        'model': {
            'name': 'landmark',
            'topk': 2048,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1280,
        },
    },
}
