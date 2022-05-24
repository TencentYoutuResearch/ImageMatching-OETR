#!/usr/bin/env python
"""
@File    :   utils.py
@Time    :   2021/07/01 11:54:04
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import math

import numpy as np
import torch
from torch import nn


def box_tlbr_to_xyxy(loc, tlbr, max_h, max_w):
    # cxywh: [N, 4]
    t, l, b, r = tlbr.unbind(-1)
    x, y = loc.unbind(-1)
    t, b = t * max_h, b * max_h
    l, r = l * max_w, r * max_w

    x1 = (x - l).clamp(min=0.0, max=max_w)
    y1 = (y - t).clamp(min=0.0, max=max_h)
    x2 = (x + r).clamp(min=0.0, max=max_w)
    y2 = (y + b).clamp(min=0.0, max=max_h)
    b = [x1, y1, x2, y2]
    return torch.stack(b, dim=-1)


def box_cxywh_to_xyxy(cxywh, max_h, max_w):
    # cxywh: [N, 4]
    cx, cy, w, h = cxywh.unbind(-1)
    x1 = (cx - w / 2).clamp(min=0.0, max=max_w)
    y1 = (cy - h / 2).clamp(min=0.0, max=max_h)
    x2 = (cx + w / 2).clamp(min=0.0, max=max_w)
    y2 = (cy + h / 2).clamp(min=0.0, max=max_h)
    b = [x1, y1, x2, y2]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxywh(xyxy, max_h, max_w):
    # cxywh: [N, 4]
    x1, y1, x2, y2 = xyxy.unbind(-1)
    x1 = x1.clamp(min=0.0, max=max_w)
    x2 = x2.clamp(min=0.0, max=max_w)
    y1 = y1.clamp(min=0.0, max=max_h)
    y2 = y2.clamp(min=0.0, max=max_h)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    b = [cx, cy, w, h]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)


def delta2bbox(
        deltas,
        means=(0.0, 0.0, 0.0, 0.0),
        stds=(1.0, 1.0, 1.0, 1.0),
        max_shape=None,
        wh_ratio_clip=16 / 1000,
):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and
    the deltas are network outputs used to shift/scale those boxes.
    This is the inverse function of `bbox2delta()`

    Args:
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 4 * num_classes). Note N = num_anchors * W * H
            when rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.

    Returns:
        Tensor: Boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Use exp(network energy) to enlarge/shrink image
    gw = dw.exp() * max_shape[1]
    gh = dh.exp() * max_shape[0]
    # Use network energy to shift the center of image
    gx = max_shape[1] * (dx + 0.5)
    gy = max_shape[0] * (dy + 0.5)

    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    x1 = x1.clamp(min=0, max=max_shape[1])
    y1 = y1.clamp(min=0, max=max_shape[0])
    x2 = x2.clamp(min=0, max=max_shape[1])
    y2 = y2.clamp(min=0, max=max_shape[0])
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def compute_locations(feat, stride=16):
    device = feat.device
    h, w = feat.shape[-2:]
    shifts_x = torch.arange(0,
                            w * stride,
                            step=stride,
                            dtype=torch.float32,
                            device=device)
    shifts_y = torch.arange(0,
                            h * stride,
                            step=stride,
                            dtype=torch.float32,
                            device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def mask2bbox(masks, stride=16, max_shape=None):
    batch_boxes = []
    for mask in masks:
        xy = mask.nonzero(as_tuple=False)
        x1 = xy[:, 0].min() * stride - stride / 2
        y1 = xy[:, 1].min() * stride - stride / 2
        x2 = xy[:, 0].max() * stride + stride / 2
        y2 = xy[:, 1].max() * stride + stride / 2
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        batch_boxes.append(bboxes)
    all_bboxes = torch.stack(batch_boxes, dim=0)
    return all_bboxes


class PositionEncodingSine(nn.Module):
    """This is a sinusoidal position encoding that generalized to 2-dimensional
    images."""
    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap,
                               the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float() *
            (-math.log(10000.0) / d_model // 2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0),
                             persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return self.pe[:, :, :x.size(2), :x.size(3)]


class PositionEmbeddingSine(nn.Module):
    """This is a more standard version of the position embedding, very similar
    to the one used by the Attention is all you need paper, generalized to work
    on images."""
    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        if mask is None:
            mask = torch.zeros_like(x)
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32,
                             device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
