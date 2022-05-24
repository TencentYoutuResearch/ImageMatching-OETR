#!/usr/bin/env python
"""
@File    :   __init__.py
@Time    :   2021/07/28 17:15:43
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""


def get_overlap(overlaper):
    mod = __import__(f'{__name__}.{overlaper}', fromlist=[''])
    return getattr(mod, 'Model')
