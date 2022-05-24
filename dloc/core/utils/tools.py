#!/usr/bin/env python
"""
@File    :   tools.py
@Time    :   2021/05/24 18:03:29
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

from collections import abc as collections

import torch
from torch._six import string_classes


def map_tensor(input_, func):
    if isinstance(input_, torch.Tensor):
        return func(input_)
    elif isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    else:
        raise TypeError(
            f'input must be tensor, dict or list; found {type(input_)}')
