#!/usr/bin/env python
'''
@File    :   losses.py
@Time    :   2021/08/02 20:29:38
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
'''
import os

import torch
from torch import nn
from torch.nn import functional as F

from .losses import SigmoidFocalLoss

INF = 1e8


def get_num_gpus():
    return int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


class IOULoss(nn.Module):
    def __init__(self, loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) +\
            torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) +\
            torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) +\
            torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


def sigmoid_focal_loss(inputs,
                       targets,
                       alpha=-1.0,
                       gamma=2.0,
                       reduction='none'):
    """
    Loss used in RetinaNet for dense detection:
    https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the
                binary classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                 targets,
                                                 reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


class FCOSLossComputation(object):
    """This class computes the FCOS losses."""
    def __init__(self,
                 stride=16,
                 center_sampling_radius=2.0,
                 iou_loss_type='giou',
                 norm_reg_targets=False):
        self.stride = stride
        self.center_sampling_radius = self.stride * center_sampling_radius
        self.iou_loss_type = iou_loss_type
        self.cls_loss_func = SigmoidFocalLoss(gamma=2.0, alpha=0.25)
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction='sum')
        self.norm_reg_targets = norm_reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *\
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def get_sample_region(self, target_bboxes, locations, ratios=16.0):
        K = target_bboxes.shape[0]
        num_points = locations.shape[0]
        center_x = (target_bboxes[:, 0] + target_bboxes[:, 2]) * 0.5
        center_y = (target_bboxes[:, 1] + target_bboxes[:, 3]) * 0.5
        xmin = torch.where(center_x - ratios > target_bboxes[:, 0],
                           center_x - ratios, target_bboxes[:, 0])
        ymin = torch.where(center_y - ratios > target_bboxes[:, 1],
                           center_y - ratios, target_bboxes[:, 1])
        xmax = torch.where(center_x + ratios < target_bboxes[:, 2],
                           center_x + ratios, target_bboxes[:, 2])
        ymax = torch.where(center_y + ratios < target_bboxes[:, 3],
                           center_y + ratios, target_bboxes[:, 3])
        # limit sample region in gt bbox
        left = locations[:, 0][None].expand(K, num_points) - xmin.unsqueeze(-1)
        right =\
            xmax.unsqueeze(-1) - locations[:, 0][None].expand(K, num_points)
        top =\
            locations[:, 1][None].expand(K, num_points) - ymin.unsqueeze(-1)
        bottom =\
            ymax.unsqueeze(-1) - locations[:, 1][None].expand(K, num_points)
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_target_for_locations(self, locations, target, stride=16.0):
        K = target.shape[0]
        num_points = locations.shape[0]
        left = locations[:, 0][None].expand(K, num_points) -\
            target[:, 0].unsqueeze(-1)
        top = locations[:, 1][None].expand(K, num_points) -\
            target[:, 1].unsqueeze(-1)
        right = target[:, 2].unsqueeze(-1) -\
            locations[:, 0][None].expand(K, num_points)
        bottom = target[:, 3].unsqueeze(-1) -\
            locations[:, 1][None].expand(K, num_points)
        reg_targets = torch.stack([left, top, right, bottom], dim=2)

        if self.center_sampling_radius > 0:
            is_in_boxes = self.get_sample_region(
                target, locations, ratios=self.center_sampling_radius)
        else:
            is_in_boxes = reg_targets.min(dim=2)[0] > 0

        if self.norm_reg_targets:
            reg_targets = reg_targets / self.stride

        return is_in_boxes, reg_targets

    def __call__(self, locations, box_cls, box_regression, centerness,
                 targets):
        """
        Arguments:
            locations: dense pixel for feature map ((HxW)x2)
            box_cls: dense box classfication only one classes(Bx1xHxW)
            box_regression: regression for overlap box (Bx4xHxW)
            centerness: pixel is center of pverlap box (Bx1xHxW)
            targets: one image with single overlap box (Bx4)

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        labels, reg_targets = self.compute_target_for_locations(
            locations, targets)

        box_cls_flatten = box_cls.permute(0, 2, 3, 1).reshape(-1, 1)
        box_regression_flatten =\
            box_regression.permute(0, 2, 3, 1).reshape(-1, 4)
        labels_flatten = labels.reshape(-1)
        reg_targets_flatten = reg_targets.reshape(-1, 4)
        centerness_flatten = centerness.reshape(-1)

        pos_inds = torch.nonzero(labels_flatten > 0, as_tuple=False).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # # sync num_pos from all gpus
        total_num_pos =\
            reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(
            box_cls_flatten, labels_flatten.long()) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            centerness_targets =\
                self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            reg_loss = self.box_reg_loss_func(
                box_regression_flatten, reg_targets_flatten,
                centerness_targets) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(
                centerness_flatten, centerness_targets) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss
