#!/usr/bin/env python
"""
@File    :   utils.py
@Time    :   2021/06/29 21:07:53
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import logging

import cv2
import numpy as np
import torch


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s]' +
        '%(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')


def to_GB(memory):
    return round(memory / 1024**3, 1)


def get_gpu_memory(name='', id=0):
    t = torch.cuda.get_device_properties(id).total_memory
    # c = torch.cuda.memory_reserved(id)
    a = torch.cuda.memory_allocated(id)
    # f = c-a  # free inside cache
    return to_GB(a), to_GB(t)
    # print('Free GPU memory : {}/{}'.format(f, t))
    # logger.info('{} GPU memory : {}/{} GB'.format(name, to_GB(a), to_GB(t)))


def process_resize(w, h, resize):
    assert len(resize) > 0 and len(resize) <= 2
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.0).float()[None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path))
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


def visualize_overlap(image1, bbox1, image2, bbox2, output):
    left = cv2.rectangle(image1, tuple(bbox1[0:2]), tuple(bbox1[2:]),
                         (255, 0, 0), 2)
    right = cv2.rectangle(image2, tuple(bbox2[0:2]), tuple(bbox2[2:]),
                          (0, 0, 255), 2)
    viz = cv2.hconcat([left, right])
    cv2.imwrite(output, viz)


def visualize_overlap_gt(image1,
                         bbox1,
                         gt1,
                         image2,
                         bbox2,
                         gt2,
                         output,
                         save=True):

    left = cv2.rectangle(image1, tuple(bbox1[0:2]), tuple(bbox1[2:]),
                         (255, 0, 0), 2)
    right = cv2.rectangle(image2, tuple(bbox2[0:2]), tuple(bbox2[2:]),
                          (0, 0, 255), 2)
    left = cv2.rectangle(left, tuple(gt1[0:2]), tuple(gt1[2:]), (0, 255, 0), 2)
    right = cv2.rectangle(right, tuple(gt2[0:2]), tuple(gt2[2:]), (0, 255, 0),
                          2)
    if save:
        viz = cv2.hconcat([left, right])
        cv2.imwrite(output, viz)
    return left, right


def visualize_centerness_overlap_gt(image1, bbox1, gt1, center1, image2, bbox2,
                                    gt2, center2, output):

    left, right = visualize_overlap_gt(image1, bbox1, gt1, image2, bbox2, gt2,
                                       output, False)
    # from IPython import embed;embed()
    center1 = (center1 - center1.min()) / center1.max()
    center2 = (center2 - center2.min()) / center2.max()
    center1 = cv2.resize(center1.astype('float32'), image1.shape[:-1]) * 255
    center2 = cv2.resize(center2.astype('float32'), image2.shape[:-1]) * 255

    center1 = cv2.applyColorMap(center1.astype(np.uint8), cv2.COLORMAP_JET)
    center2 = cv2.applyColorMap(center2.astype(np.uint8), cv2.COLORMAP_JET)

    left = cv2.addWeighted(left, 1.0, center1.astype('float32'), 0.4, 0)
    right = cv2.addWeighted(right, 1.0, center2.astype('float32'), 0.4, 0)

    viz = cv2.hconcat([left, right])
    cv2.imwrite(output, viz)


def visualization_heatmap():
    pass


def loss_info(infos, writer, iter):
    str_info = ''
    for k, v in infos.items():
        if 'loss' in k:
            str_info += '{}:{:.5f}, '.format(k, v)
            writer.add_scalar('Loss/{}'.format(k), v, iter)
        if 'iou' in k:
            str_info += '{}:{:.5f}, '.format(k, v)
            writer.add_scalar('iou/{}'.format(k), v, iter)
    return str_info
