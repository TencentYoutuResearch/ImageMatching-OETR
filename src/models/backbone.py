#!/usr/bin/env python
'''
@File    :   encoder.py
@Time    :   2021/06/29 16:22:59
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
'''

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from einops.einops import rearrange
from timm.models.layers import to_2tuple


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.
        Default: nn.LayerNorm
    """
    def __init__(self,
                 input_resolution,
                 dim,
                 norm_layer=nn.LayerNorm,
                 patch_size=[2]):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size
        self.norm = norm_layer(dim)

        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = 2 * dim // 2**i
            else:
                out_dim = 2 * dim // 2**(i + 1)
            stride = 2
            padding = (ps - stride) // 2
            self.reductions.append(
                nn.Conv2d(dim,
                          out_dim,
                          kernel_size=ps,
                          stride=stride,
                          padding=padding))

    def forward(self, x):  # , H, W):
        """
        x: B, H*W, C
        """
        b, c, h, w = x.shape
        x = rearrange(x, 'n c h w -> n (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'n (h w) c -> n c h w', h=h, w=w).contiguous()

        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x)
            xs.append(tmp_x)
        x = torch.cat(xs, dim=1)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels.
        Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self,
                 img_size=224,
                 patch_size=[4],
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patches_resolution = [img_size[0] // 4, img_size[1] // 4]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.projs = nn.ModuleList()
        # i:    0       1       2     3      4 8 16 32
        # dim:  d//2   d//4   d//8  d//8
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2**i
            else:
                dim = embed_dim // 2**(i + 1)
            stride = 4
            padding = (ps - 4) // 2
            self.projs.append(
                nn.Conv2d(in_chans,
                          dim,
                          kernel_size=ps,
                          stride=stride,
                          padding=padding))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x).flatten(2).transpose(1, 2)
            xs.append(tx)  # B Ph*Pw C
        x = torch.cat(xs, dim=2)
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W


class ResnetEncoder(nn.Module):
    def __init__(self, cfg):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.cfg = cfg
        self.last_layer = cfg.BACKBONE.LAST_LAYER

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152
        }
        # pdb.set_trace()
        encoder = resnets[cfg.BACKBONE.NUM_LAYERS](True)
        self.encoder = encoder

        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        if cfg.BACKBONE.LAYER == 'layer4':
            self.layer4 = encoder.layer4
        del encoder

        if cfg.BACKBONE.NUM_LAYERS > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        x = input_image.permute(0, 3, 1, 2).contiguous()
        # Normalize the input colorspace
        if self.cfg.NORM_INPUT:
            x = (x - 0.45) / 0.225

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.cfg.BACKBONE.LAYER == 'layer3':
            x = self.layer3(x)
        elif self.cfg.BACKBONE.LAYER == 'layer4':
            x = self.layer3(x)
            x = self.layer4(x)

        return x
