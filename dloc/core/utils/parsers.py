#!/usr/bin/env python
"""
@File    :   parsers.py
@Time    :   2021/04/24 16:50:27
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np


# Get intrinsic and image infos from paths
def parse_image_lists_with_intrinsics(paths):
    results = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0

    for lfile in files:
        with open(lfile, 'r') as f:
            raw_data = f.readlines()

        logging.info(f'Importing {len(raw_data)} queries in {lfile.name}')
        for data in raw_data:
            data = data.strip('\n').split(' ')
            name, camera_model, width, height = data[:4]
            params = np.array(data[4:], float)
            info = (camera_model, int(width), int(height), params)
            results.append((name, info))

    assert len(results) > 0
    return results


# Get all retrieval files from path file
def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            q, r = p.split(' ')
            retrieval[q].append(r)
    return retrieval


# combine two names to a single pair
def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))
