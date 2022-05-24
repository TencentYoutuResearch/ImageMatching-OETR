#!/usr/bin/env python
'''
@File    :   validation.py
@Time    :   2021/06/30 17:08:35
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
'''
import os

import numpy as np
import torch
from tqdm import tqdm

from src.losses.utils import bbox_oiou, bbox_overlaps

from .utils import visualize_centerness_overlap_gt, visualize_overlap_gt


def _recalls(ious, thrs):
    img_num = ious.shape[0]
    recalls = np.zeros(thrs.size)
    for i, thr in enumerate(thrs):
        recalls[i] = (ious >= thr).sum() / float(img_num)
    return recalls


def eval_recalls(ious, iou_thrs=0.5, logger=None):
    """Calculate recalls.

    Args:
        ious (list[ndarray]): a list of arrays of shape (n, 4)
        iou_thrs (float | Sequence[float]): IoU thresholds. Default: 0.5.
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        ndarray: recalls of different ious and proposal nums
    """
    recalls = _recalls(np.array(ious), np.array(iou_thrs))
    if logger:
        logger.info('Validation results:')
        logger.info('Recalls\t R0.5\t R0.75\t R0.9\t')
        logger.info('Values\t {:.5f}\t {:.5f}\t {:.5f}\t'.format(
            recalls[0], recalls[5], recalls[8]))
    else:
        print('Recalls\t R0.5\t R0.75\t R0.9\t')
        print('Values\t {:.5f}\t {:.5f}\t {:.5f}\t'.format(
            recalls[0], recalls[5], recalls[8]))
    return recalls


@torch.no_grad()
def evaluate(model,
             dataloader,
             logger,
             save_path,
             iou_thrs=np.arange(0.5, 0.96, 0.05),
             epoch=0,
             oiou=False,
             viz=False):
    ious = []
    oious = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        data = model(batch, validation=True)

        if not oiou:
            ious1 = bbox_overlaps(batch['overlap_box1'],
                                  data['pred_bbox1'].cpu(),
                                  is_aligned=True)
            ious2 = bbox_overlaps(batch['overlap_box2'],
                                  data['pred_bbox2'].cpu(),
                                  is_aligned=True)
        else:
            ious1 = bbox_overlaps(batch['overlap_box1'],
                                  data['pred_bbox1'].cpu(),
                                  is_aligned=True)
            ious2 = bbox_overlaps(batch['overlap_box2'],
                                  data['pred_bbox2'].cpu(),
                                  is_aligned=True)
            oious1 = bbox_oiou(batch['overlap_box1'], data['pred_bbox1'].cpu())
            oious2 = bbox_oiou(batch['overlap_box2'], data['pred_bbox2'].cpu())
            oious += list(oious1.numpy()) + list(oious2.numpy())
        ious += list(ious1.numpy()) + list(ious2.numpy())

        if i % 10 == 0 and viz:
            bbox1 = data['pred_bbox1'][0].cpu().numpy().astype(int)
            bbox2 = data['pred_bbox2'][0].cpu().numpy().astype(int)
            gt_bbox1 = batch['overlap_box1'][0].numpy().astype(int)
            gt_bbox2 = batch['overlap_box2'][0].numpy().astype(int)
            viz_name = os.path.join(
                str(save_path),
                'epoch' + str(epoch) + '_' + batch['file_name'][0])
            if 'pred_center1' in data.keys():
                visualize_centerness_overlap_gt(
                    batch['image1'][0].cpu().numpy() * 255, bbox1, gt_bbox1,
                    data['pred_center1'][0].cpu().numpy(),
                    batch['image2'][0].cpu().numpy() * 255, bbox2, gt_bbox2,
                    data['pred_center2'][0].cpu().numpy(), viz_name)
            else:
                visualize_overlap_gt(batch['image1'][0].cpu().numpy() * 255,
                                     bbox1, gt_bbox1,
                                     batch['image2'][0].cpu().numpy() * 255,
                                     bbox2, gt_bbox2, viz_name)

    eval_recalls(ious, iou_thrs, logger)
    if oiou:
        eval_recalls(oious, iou_thrs, logger)


@torch.no_grad()
def evaluate_dummy(model,
                   dataloader,
                   logger,
                   save_path,
                   iou_thrs=np.arange(0.5, 0.96, 0.05),
                   epoch=0,
                   oiou=False,
                   viz=False):
    ious = []
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image1, image2 = batch['image1'].cuda(), batch['image2'].cuda()
        pred_bbox1, pred_bbox2 = model.forward_dummy(image1, image2)
        if not oiou:
            ious1 = bbox_overlaps(batch['overlap_box1'],
                                  pred_bbox1.cpu(),
                                  is_aligned=True)
            ious2 = bbox_overlaps(batch['overlap_box2'],
                                  pred_bbox2.cpu(),
                                  is_aligned=True)
        else:
            ious1 = bbox_oiou(batch['overlap_box1'], pred_bbox1.cpu())
            ious2 = bbox_oiou(batch['overlap_box2'], pred_bbox2.cpu())
        ious += list(ious1.numpy()) + list(ious2.numpy())
        if viz:
            bbox1 = pred_bbox1[0].cpu().numpy().astype(int)
            bbox2 = pred_bbox2[0].cpu().numpy().astype(int)
            gt_bbox1 = batch['overlap_box1'][0].numpy().astype(int)
            gt_bbox2 = batch['overlap_box2'][0].numpy().astype(int)
            viz_name = os.path.join(
                str(save_path),
                'epoch' + str(epoch) + '_' + batch['file_name'][0])
            visualize_overlap_gt(batch['image1'][0].numpy() * 255, bbox1,
                                 gt_bbox1, batch['image2'][0].numpy() * 255,
                                 bbox2, gt_bbox2, viz_name)
    eval_recalls(ious, iou_thrs, logger)
