#!/usr/bin/env python
"""
@File    :   cotr.py
@Time    :   2021/04/25 13:04:05
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""


def get_matcher(matcher):
    mod = __import__(f'{__name__}.{matcher}', fromlist=[''])
    return getattr(mod, 'Model')
