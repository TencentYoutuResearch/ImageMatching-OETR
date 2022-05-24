#!/usr/bin/env python
"""
@File    :   trainner.py
@Time    :   2021/06/29 19:21:04
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from kornia.utils import create_meshgrid

from .losses.losses import CycleOverlapLoss, IouOverlapLoss
from .losses.utils import bbox_oiou, bbox_overlaps
from .models.backbone import PatchMerging, ResnetEncoder
from .models.transformer import QueryTransformer
from .models.utils import (PositionEncodingSine, box_tlbr_to_xyxy,
                           box_xyxy_to_cxywh)

INF = 1e9


def MLP(channels, do_bn=True):
    """Multi-layer perceptron."""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class OETR(nn.Module):
    """OETR model architecture."""
    def __init__(self, cfg):
        super(OETR, self).__init__()
        # Architecture of feature extraction
        self.backbone = ResnetEncoder(cfg)
        self.d_model = self.backbone.last_layer // 4
        self.input_proj = nn.Conv2d(self.backbone.last_layer,
                                    self.d_model,
                                    kernel_size=1)
        self.input_proj2 = nn.Conv2d(self.d_model * 2,
                                     self.d_model,
                                     kernel_size=1)
        self.patchmerging = PatchMerging(
            (20, 20),
            self.d_model,
            norm_layer=nn.LayerNorm,
            patch_size=[4, 8, 16],
        )

        # Regression module
        self.tlbr_reg = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, False),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 4),
        )

        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(
                self.d_model,
                self.d_model,
                (3, 3),
                padding=(1, 1),
                stride=(1, 1),
                bias=True,
            ),
            nn.GroupNorm(32, self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model, 1, (1, 1)),
        )
        # Only one overlap for every image, so query is set to one
        num_queries = 1
        self.query_embed1 = nn.Embedding(num_queries, self.d_model)
        self.query_embed2 = nn.Embedding(num_queries, self.d_model)
        self.transformer = QueryTransformer(self.d_model,
                                            nhead=8,
                                            num_layers=4)
        self.pos_encoding = PositionEncodingSine(self.d_model,
                                                 max_shape=cfg.NECK.MAX_SHAPE)

        # Loss module
        self.iouloss = IouOverlapLoss(reduction='mean', oiou=cfg.LOSS.OIOU)
        self.cycle_loss = CycleOverlapLoss()

        # Hyperparameters
        self.max_shape = cfg.NECK.MAX_SHAPE
        self.cycle = cfg.LOSS.CYCLE_OVERLAP
        self.softmax_temperature = 1

        # self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.fc_reg.weight, 0, 0.001)
        nn.init.constant_(self.fc_reg.bias, 0)

    def generate_mesh_grid(self, feat_hw, stride, device='cpu'):
        """generate mesh grid with specific width, height and stride."""
        coord_xy_map = (create_meshgrid(feat_hw[0], feat_hw[1], False, device)
                        + 0.5) * stride
        return coord_xy_map.reshape(1, feat_hw[0] * feat_hw[1], 2)

    def feature_extraction(self, image1, image2, mask1=None, mask2=None):
        """forward image pairs overlap estimation."""
        # N, h1, w1, _ = image1.shape
        # h2, w2 = image2.shape[1:3]

        feat1 = self.backbone(image1)
        feat2 = self.backbone(image2)
        feat1 = self.input_proj(feat1)
        feat2 = self.input_proj(feat2)

        feat1 = self.patchmerging(feat1)
        feat2 = self.patchmerging(feat2)
        feat1 = self.input_proj2(feat1)
        feat2 = self.input_proj2(feat2)

        hf1, wf1 = feat1.shape[2:]
        hf2, wf2 = feat2.shape[2:]

        pos1 = self.pos_encoding(feat1)
        pos2 = self.pos_encoding(feat2)

        return feat1, feat2, pos1, pos2, hf1, wf1, hf2, wf2

    def feature_correlation(self, feat1, feat2, pos1, pos2, mask1, mask2):
        hs1, hs2, memory1, memory2 = self.transformer(
            feat1,
            feat2,
            self.query_embed1.weight,
            self.query_embed2.weight,
            pos1,
            pos2,
            mask1,
            mask2,
        )
        return hs1, hs2, memory1, memory2

    def center_estimation(self, hs1, hs2, memory1, memory2, hf1, wf1, hf2, wf2,
                          mask1, mask2):
        att1 = torch.einsum('blc, bnc->bln', memory1,
                            hs1)  # [N, hw, num_q]  num_q=1
        att2 = torch.einsum('blc, bnc->bln', memory2, hs2)

        # weighted sum for center regression
        heatmap1 = rearrange(memory1 * att1,
                             'n (h w) c -> n c h w',
                             h=hf1,
                             w=wf1)
        heatmap2 = rearrange(memory2 * att2,
                             'n (h w) c -> n c h w',
                             h=hf2,
                             w=wf2)
        heatmap1_flatten = (
            rearrange(self.heatmap_conv(heatmap1), 'n c h w -> n (h w) c') *
            self.softmax_temperature)
        heatmap2_flatten = (
            rearrange(self.heatmap_conv(heatmap2), 'n c h w -> n (h w) c') *
            self.softmax_temperature)

        if mask1 is not None:
            heatmap1_flatten.masked_fill_(~mask1.flatten(1)[..., None].bool(),
                                          -INF)
            heatmap2_flatten.masked_fill_(~mask2.flatten(1)[..., None].bool(),
                                          -INF)

        prob_map1 = nn.functional.softmax(heatmap1_flatten,
                                          dim=1)  # [N, hw, 1]
        prob_map2 = nn.functional.softmax(heatmap2_flatten, dim=1)
        coord_xy_map1 = self.generate_mesh_grid(
            (hf1, wf1), stride=self.h1 // hf1,
            device=memory1.device)  # [1, h*w, 2]   # .repeat(N, 1, 1, 1)
        coord_xy_map2 = self.generate_mesh_grid(
            (hf2, wf2), stride=self.h2 // hf2,
            device=memory2.device)  # .repeat(N, 1, 1, 1)

        box_cxy1 = (prob_map1 * coord_xy_map1).sum(1)  # [N, 2]
        box_cxy2 = (prob_map2 * coord_xy_map2).sum(1)

        return box_cxy1, box_cxy2

    def size_regression(self, hs1, hs2):
        tlbr1 = self.tlbr_reg(hs1).sigmoid().squeeze(1)
        tlbr2 = self.tlbr_reg(hs2).sigmoid().squeeze(1)
        return tlbr1, tlbr2

    def obtain_overlap_bbox(self, box_cxy1, tlbr1, box_cxy2, tlbr2):
        pred_bbox_xyxy1 = torch.stack(
            [
                box_cxy1[:, 0] - tlbr1[:, 1] * self.w1,
                box_cxy1[:, 1] - tlbr1[:, 0] * self.h1,
                box_cxy1[:, 0] + tlbr1[:, 3] * self.w1,
                box_cxy1[:, 1] + tlbr1[:, 2] * self.h1,
            ],
            dim=1,
        )
        pred_bbox_xyxy2 = torch.stack(
            [
                box_cxy2[:, 0] - tlbr2[:, 1] * self.w2,
                box_cxy2[:, 1] - tlbr2[:, 0] * self.h2,
                box_cxy2[:, 0] + tlbr2[:, 3] * self.w2,
                box_cxy2[:, 1] + tlbr2[:, 2] * self.h2,
            ],
            dim=1,
        )
        pred_bbox_cxywh1 = torch.cat(
            [
                (pred_bbox_xyxy1[:, :2] + pred_bbox_xyxy1[:, 2:]) / 2,
                pred_bbox_xyxy1[:, 2:] - pred_bbox_xyxy1[:, :2],
            ],
            dim=-1,
        )
        pred_bbox_cxywh2 = torch.cat(
            [
                (pred_bbox_xyxy2[:, :2] + pred_bbox_xyxy2[:, 2:]) / 2,
                pred_bbox_xyxy2[:, 2:] - pred_bbox_xyxy2[:, :2],
            ],
            dim=-1,
        )
        return pred_bbox_xyxy1, pred_bbox_xyxy2, pred_bbox_cxywh1, pred_bbox_cxywh2

    # Inference pipeline
    def forward_dummy(self, image1, image2, mask1=None, mask2=None):
        h1, w1 = image1.shape[1:3]
        h2, w2 = image2.shape[1:3]
        self.h1, self.w1 = h1, w1
        self.h2, self.w2 = h2, w2

        # feature extraction
        feat1, feat2, pos1, pos2, hf1, wf1, hf2, wf2 = self.feature_extraction(
            image1, image2, mask1, mask2)

        # feature correlation
        hs1, hs2, memory1, memory2 = self.feature_correlation(
            feat1, feat2, pos1, pos2, mask1, mask2)

        # overlap regression
        box_cxy1, box_cxy2 = self.center_estimation(hs1, hs2, memory1, memory2,
                                                    hf1, wf1, hf2, wf2, mask1,
                                                    mask2)
        tlbr1, tlbr2 = self.size_regression(hs1, hs2)

        pred_bbox_xyxy1 = box_tlbr_to_xyxy(box_cxy1, tlbr1, max_h=h1, max_w=w1)
        pred_bbox_xyxy2 = box_tlbr_to_xyxy(box_cxy2, tlbr2, max_h=h2, max_w=w2)

        return pred_bbox_xyxy1, pred_bbox_xyxy2

    # Trainning pipeline
    def forward(self, data, validation=False):
        if 'resize_mask1' in data:
            mask1 = data['resize_mask1'][data['overlap_valid']]
            mask2 = data['resize_mask2'][data['overlap_valid']]
        else:
            mask1, mask2 = None, None
        h1, w1 = data['image1'][data['overlap_valid']].shape[1:3]
        h2, w2 = data['image2'][data['overlap_valid']].shape[1:3]
        self.h1, self.w1 = h1, w1
        self.h2, self.w2 = h2, w2

        # feature extraction
        feat1, feat2, pos1, pos2, hf1, wf1, hf2, wf2 = self.feature_extraction(
            data['image1'][data['overlap_valid']],
            data['image2'][data['overlap_valid']],
            mask1,
            mask2,
        )

        # feature correlation
        hs1, hs2, memory1, memory2 = self.feature_correlation(
            feat1, feat2, pos1, pos2, mask1, mask2)

        # overlap regression
        box_cxy1, box_cxy2 = self.center_estimation(hs1, hs2, memory1, memory2,
                                                    hf1, wf1, hf2, wf2, mask1,
                                                    mask2)
        tlbr1, tlbr2 = self.size_regression(hs1, hs2)

        (
            pred_bbox_xyxy1,
            pred_bbox_xyxy2,
            pred_bbox_cxywh1,
            pred_bbox_cxywh2,
        ) = self.obtain_overlap_bbox(box_cxy1, tlbr1, box_cxy2, tlbr2)

        # Groundtruth
        gt_bbox_xyxy1 = data['overlap_box1'][data['overlap_valid']]
        gt_bbox_xyxy2 = data['overlap_box2'][data['overlap_valid']]
        gt_bbox_cxywh1 = box_xyxy_to_cxywh(gt_bbox_xyxy1, max_h=h1, max_w=w1)
        gt_bbox_cxywh2 = box_xyxy_to_cxywh(gt_bbox_xyxy2, max_h=h2, max_w=w2)

        wh_scale1 = torch.tensor([w1, h1], device=data['image1'].device)
        wh_scale2 = torch.tensor([w2, h2], device=data['image2'].device)
        # Localization loss
        loc_l1_loss = F.l1_loss(
            pred_bbox_cxywh1[:, :2] / wh_scale1,
            gt_bbox_cxywh1[:, :2] / wh_scale1,
            reduction='mean',
        ) + F.l1_loss(
            pred_bbox_cxywh2[:, :2] / wh_scale2,
            gt_bbox_cxywh2[:, :2] / wh_scale2,
            reduction='mean',
        )

        # Width and height loss
        wh_l1_loss = (F.l1_loss(
            pred_bbox_cxywh1[:, 2:] / wh_scale1,
            gt_bbox_cxywh1[:, 2:] / wh_scale1,
            reduction='mean',
        ) + F.l1_loss(
            pred_bbox_cxywh2[:, 2:] / wh_scale2,
            gt_bbox_cxywh2[:, 2:] / wh_scale2,
            reduction='mean',
        )) / 2

        # Custom iou loss
        iouloss = self.iouloss(pred_bbox_xyxy1, gt_bbox_xyxy1, pred_bbox_xyxy2,
                               gt_bbox_xyxy2)

        # IOU results
        iou1 = bbox_overlaps(
            pred_bbox_xyxy1,
            data['overlap_box1'][data['overlap_valid']],
            is_aligned=True,
        ).mean()
        iou2 = bbox_overlaps(
            pred_bbox_xyxy2,
            data['overlap_box2'][data['overlap_valid']],
            is_aligned=True,
        ).mean()
        oiou1 = bbox_oiou(data['overlap_box1'][data['overlap_valid']],
                          pred_bbox_xyxy1).mean()
        oiou2 = bbox_oiou(data['overlap_box2'][data['overlap_valid']],
                          pred_bbox_xyxy2).mean()

        results = {
            'pred_bbox1': pred_bbox_xyxy1,
            'pred_bbox2': pred_bbox_xyxy2,
            'iouloss': iouloss.mean(),
            'wh_loss': wh_l1_loss.mean(),
            'loc_loss': loc_l1_loss.mean(),
            'iou1': iou1,
            'iou2': iou2,
            'oiou1': oiou1,
            'oiou2': oiou2,
        }

        # Using cycle consistency loss
        if self.cycle:
            box_cxy1from2, box_cxy2from1 = self.center_estimation(
                hs2, hs1, memory1, memory2, hf1, wf1, hf2, wf2, mask1, mask2)
            (
                _,
                _,
                pred_bbox_cxywh1from2,
                pred_bbox_cxywh2from1,
            ) = self.obtain_overlap_bbox(box_cxy1from2, tlbr1, box_cxy2from1,
                                         tlbr2)
            cycle_loss = F.l1_loss(
                pred_bbox_cxywh1from2[:, :2] / wh_scale1,
                gt_bbox_cxywh1[:, :2] / wh_scale1,
                reduction='mean',
            ) + F.l1_loss(
                pred_bbox_cxywh2from1[:, :2] / wh_scale2,
                gt_bbox_cxywh2[:, :2] / wh_scale2,
                reduction='mean',
            )

            results['cycle_loss'] = cycle_loss.mean()

        return results


# build model with different configuration
def build_detectors(cfg):
    if cfg.MODEL == 'oetr':
        return OETR(cfg)
    else:
        raise ValueError(f'OETR.MODEL {cfg.MODEL} not supported.')
