#!/usr/bin/env python
"""
@File    :   losses.py
@Time    :   2021/06/29 16:52:52
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import bbox_oiou, bbox_overlaps


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(1,
                                 n_class + 1,
                                 dtype=target.dtype,
                                 device=target.device).unsqueeze(0)

        t = target.unsqueeze(1).contiguous()
        p = torch.sigmoid(out)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p)**gamma * torch.log(p)
        term2 = p**gamma * torch.log(1 - p)

        # print(term1.sum(), term2.sum())

        loss = (-(t == class_ids).float() * alpha * term1 -
                ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2)

        return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.unsqueeze(1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def iou_loss(pred, target, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        torch.Tensor: Loss tensor.
    """
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    loss = -ious.log()
    return loss


def oiou_loss(pred, target, eps=1e-7):
    ious = bbox_oiou(target, pred, eps)
    loss = 1 - ious
    return loss


def giou_loss(pred, target, eps=1e-7):
    """Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression https://arxiv.org/abs/1902.09630.

    code refer to:
    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps

    # GIoU
    gious = ious - (enclose_area - union) / enclose_area
    loss = 1 - gious
    return loss


class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if ((weight is not None) and (not torch.any(weight > 0))
                and (reduction != 'none')):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * iou_loss(pred, target, eps=self.eps)
        return loss


class IouOverlapLoss(nn.Module):
    """OverlapLoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """
    def __init__(self,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 oiou=False):
        super(IouOverlapLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.oiou = oiou

    def forward(self,
                pred1,
                target1,
                pred2,
                target2,
                weight=None,
                reduction_override=None,
                **kwargs):
        if self.oiou:
            loss1 = oiou_loss(pred1, target1)
            loss2 = oiou_loss(pred2, target2)
        else:
            loss1 = giou_loss(pred1, target1)
            loss2 = giou_loss(pred2, target2)
        return (loss1 + loss2) / 2.0


class MaskOverlapLoss(nn.Module):
    def __init__(self):
        super(MaskOverlapLoss, self).__init__()

    def forward(self):
        pass


class CycleOverlapLoss(nn.Module):
    def __init__(self):
        super(CycleOverlapLoss, self).__init__()

    def depth_box_filter(self, depth, bbox):
        depth[torch.isnan(depth)] = 0
        valid_depth_xy = (depth > 0).nonzero(as_tuple=False)
        if valid_depth_xy.shape[0] == 0:
            return valid_depth_xy.transpose(0, 1)
        valid = torch.min(
            torch.min(valid_depth_xy[:, 1] > bbox[0],
                      valid_depth_xy[:, 0] > bbox[1]),
            torch.min(valid_depth_xy[:, 1] < bbox[2],
                      valid_depth_xy[:, 0] < bbox[3]),
        )
        valid_box_xy = valid_depth_xy[valid]
        valid_box_xyz = torch.cat(
            (
                valid_box_xy[:, [1, 0]],
                depth[valid_box_xy[:, 0], valid_box_xy[:, 1]].reshape(-1, 1),
            ),
            dim=1,
        )
        return valid_box_xyz.transpose(0, 1)

    def interpolate_depth(self, pos, depth):
        ids = torch.arange(0, pos.size(1))
        h, w = depth.size()

        i = pos[1, :]
        j = pos[0, :]

        # Valid corners
        i_top_left = torch.floor(i).long()
        j_top_left = torch.floor(j).long()
        valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

        i_top_right = torch.floor(i).long()
        j_top_right = torch.ceil(j).long()
        valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

        i_bottom_left = torch.ceil(i).long()
        j_bottom_left = torch.floor(j).long()
        valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

        i_bottom_right = torch.ceil(i).long()
        j_bottom_right = torch.ceil(j).long()
        valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

        valid_corners = torch.min(
            torch.min(valid_top_left, valid_top_right),
            torch.min(valid_bottom_left, valid_bottom_right),
        )

        i_top_left = i_top_left[valid_corners]
        j_top_left = j_top_left[valid_corners]

        i_top_right = i_top_right[valid_corners]
        j_top_right = j_top_right[valid_corners]

        i_bottom_left = i_bottom_left[valid_corners]
        j_bottom_left = j_bottom_left[valid_corners]

        i_bottom_right = i_bottom_right[valid_corners]
        j_bottom_right = j_bottom_right[valid_corners]

        ids = ids[valid_corners]
        if ids.size(0) == 0:
            return [ids, ids, ids]

        # Valid depth
        valid_depth = torch.min(
            torch.min(depth[i_top_left, j_top_left] > 0,
                      depth[i_top_right, j_top_right] > 0),
            torch.min(
                depth[i_bottom_left, j_bottom_left] > 0,
                depth[i_bottom_right, j_bottom_right] > 0,
            ),
        )

        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]

        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]

        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]

        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]

        ids = ids[valid_depth]
        if ids.size(0) == 0:
            return [ids, ids, ids]

        # Interpolation
        i = i[ids]
        j = j[ids]
        dist_i_top_left = i - i_top_left.float()
        dist_j_top_left = j - j_top_left.float()
        w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
        w_top_right = (1 - dist_i_top_left) * dist_j_top_left
        w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
        w_bottom_right = dist_i_top_left * dist_j_top_left

        interpolated_depth = (
            w_top_left * depth[i_top_left, j_top_left] +
            w_top_right * depth[i_top_right, j_top_right] +
            w_bottom_left * depth[i_bottom_left, j_bottom_left] +
            w_bottom_right * depth[i_bottom_right, j_bottom_right])

        pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

        return [interpolated_depth, pos, ids]

    def warp(
        self,
        coord,
        K1,
        pose1,
        bbox1,
        ratio1,
        K2,
        depth2,
        pose2,
        bbox2,
        ratio2,
        max_shape,
    ):
        u1 = (coord[0, :] + bbox1[1] + 0.5) / ratio1[1]
        v1 = (coord[1, :] + bbox1[0] + 0.5) / ratio1[0]

        X1 = (u1 - K1[0, 2]) * (coord[2, :] / K1[0, 0])
        Y1 = (v1 - K1[1, 2]) * (coord[2, :] / K1[1, 1])

        XYZ1_hom = torch.cat(
            [
                X1.view(1, -1),
                Y1.view(1, -1),
                coord[2, :].view(1, -1),
                torch.ones_like(X1.view(1, -1)),
            ],
            dim=0,
        )
        XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom)
        XYZ2 = XYZ2_hom[:-1, :] / XYZ2_hom[-1, :].view(1, -1)

        uv2_hom = torch.matmul(K2, XYZ2)
        uv2 = uv2_hom[:-1, :] / uv2_hom[-1, :].view(1, -1)

        u2 = uv2[0, :] * ratio2[1] - bbox2[1] - 0.5
        v2 = uv2[1, :] * ratio2[0] - bbox2[0] - 0.5
        uv2 = torch.cat([u2.view(1, -1), v2.view(1, -1)], dim=0)
        # depth check
        ann_z2, _, ids = self.interpolate_depth(uv2, depth2)
        estimate_depth = XYZ2[2, ids]
        uv2 = uv2[:, ids]
        inlier_mask = torch.abs(estimate_depth -
                                ann_z2.to(estimate_depth.device)) < 1.0
        uv2 = uv2[:, inlier_mask]
        if uv2.shape[1] == 0:
            return uv2, uv2

        bx1 = uv2[0, :].min()
        bx2 = uv2[0, :].max()
        by1 = uv2[1, :].min()
        by2 = uv2[1, :].max()
        if max_shape is not None:
            bx1 = bx1.clamp(min=0, max=max_shape[1])
            by1 = by1.clamp(min=0, max=max_shape[0])
            bx2 = bx2.clamp(min=0, max=max_shape[1])
            by2 = by2.clamp(min=0, max=max_shape[0])
        bboxes = torch.stack([bx1, by1, bx2, by2], dim=-1)
        return uv2, bboxes

    def visualize_warped(self, image, target, pred_bbox, warped_bbox, coord):
        viz = image.cpu().numpy() * 255
        cv2.rectangle(
            viz,
            tuple(pred_bbox.detach().cpu().numpy()[0:2]),
            tuple(pred_bbox.detach().cpu().numpy()[2:]),
            (255, 0, 0),
            2,
        )
        cv2.rectangle(
            viz,
            tuple(warped_bbox.cpu().numpy()[0:2]),
            tuple(warped_bbox.cpu().numpy()[2:]),
            (0, 0, 255),
            2,
        )
        cv2.rectangle(
            viz,
            tuple(target.cpu().numpy()[0:2]),
            tuple(target.cpu().numpy()[2:]),
            (0, 255, 0),
            2,
        )
        mask = np.zeros((image.shape), dtype=np.float32)
        for i in range(coord.shape[1]):
            mask = cv2.circle(mask, (coord[0, i], coord[1, i]), 1,
                              (255, 255, 255))
        viz = cv2.addWeighted(viz, 0.7, mask, 0.3, 0)
        return viz

    def visualize_pair(
        self,
        image1,
        target1,
        pred_bbox1,
        warped_bbox1,
        coord1,
        image2,
        target2,
        pred_bbox2,
        warped_bbox2,
        coord2,
        output,
    ):
        left = self.visualize_warped(image1, target1, pred_bbox1, warped_bbox1,
                                     coord1)
        right = self.visualize_warped(image2, target2, pred_bbox2,
                                      warped_bbox2, coord2)
        viz = cv2.hconcat([left, right])
        cv2.imwrite('cycle_' + output, viz)

    def forward(
        self,
        image1,
        target1,
        pred1,
        depth1,
        K1,
        pose1,
        bbox1,
        ratio1,
        shape1,
        image2,
        target2,
        pred2,
        depth2,
        K2,
        pose2,
        bbox2,
        ratio2,
        shape2,
        output,
    ):
        warped_bbox1 = []
        warped_bbox2 = []
        valid = torch.tensor([True] * pred1.shape[0])
        for i in range(pred1.shape[0]):
            valid_box_xy1 = self.depth_box_filter(depth1[i], pred1[i])
            valid_box_xy2 = self.depth_box_filter(depth2[i], pred2[i])
            if valid_box_xy1.shape[1] == 0 or valid_box_xy2.shape[1] == 0:
                valid[i] = False
                continue
            coord2, warped2 = self.warp(
                valid_box_xy1,
                K1[i],
                pose1[i],
                bbox1[i],
                ratio1[i],
                K2[i],
                depth2[i],
                pose2[i],
                bbox2[i],
                ratio2[i],
                shape2,
            )
            coord1, warped1 = self.warp(
                valid_box_xy2,
                K2[i],
                pose2[i],
                bbox2[i],
                ratio2[i],
                K1[i],
                depth1[i],
                pose1[i],
                bbox1[i],
                ratio1[i],
                shape1,
            )
            if coord1.shape[1] == 0 or coord2.shape[1] == 0:
                valid[i] = False
                continue

            warped_bbox1.append(warped1)
            warped_bbox2.append(warped2)
        warped_bbox1 = torch.stack(warped_bbox1, dim=0)
        warped_bbox2 = torch.stack(warped_bbox2, dim=0)
        loss1 = giou_loss(pred1[valid], warped_bbox1)
        loss2 = giou_loss(pred2[valid], warped_bbox2)
        return (loss1 + loss2) / 2.0
