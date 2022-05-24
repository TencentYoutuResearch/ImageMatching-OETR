#!/usr/bin/env python
"""
@File    :   transformer.py
@Time    :   2021/07/01 15:24:40
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .linear_attention import FullAttention, LinearAttention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        kdim=None,
        vdim=None,
        attention='linear',
    ):
        super(MultiHeadAttention, self).__init__()
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.nhead = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert (self.head_dim * num_heads == self.embed_dim
                ), 'embed_dim must be divisible by num_heads'

        # multi-head attention
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        if attention == 'linear':
            self.attention = LinearAttention()
        else:
            self.attention = FullAttention()
        self.merge = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, q, k, v, q_mask=None, kv_mask=None):
        bs = q.size(0)
        # multi-head attention
        # [N, L, (H, D)]
        query = self.q_proj(q).view(bs, -1, self.nhead, self.head_dim)
        # [N, S, (H, D)]
        key = self.k_proj(k).view(bs, -1, self.nhead, self.head_dim)
        value = self.v_proj(v).view(bs, -1, self.nhead, self.head_dim)
        # [N, L, (H, D)]
        message = self.attention(query,
                                 key,
                                 value,
                                 q_mask=q_mask,
                                 kv_mask=kv_mask)
        # [N, L, C]
        message = self.merge(message.view(bs, -1, self.nhead * self.head_dim))

        return message


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention='linear'):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        if attention == 'linear':
            self.attention = LinearAttention()
        else:
            self.attention = FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.pre_norm_q = nn.LayerNorm(d_model)
        self.pre_norm_kv = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,
                x,
                source,
                x_mask=None,
                source_mask=None,
                x_pos=None,
                s_pos=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        # pdb.set_trace()
        bs = x.size(0)
        query = self.pre_norm_q(x)
        key = self.pre_norm_kv(source)
        value = self.pre_norm_kv(source)
        if x_pos is not None:
            query = query + x_pos
            key = key + s_pos
            value = value + s_pos

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query,
                                 key,
                                 value,
                                 q_mask=x_mask,
                                 kv_mask=source_mask)
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))

        # feed-forward network
        x = x + message
        message2 = self.mlp(self.norm2(x))
        return x + message2


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer module."""
    def __init__(self, d_model=512, nhead=8):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = ['self', 'cross'] * 4
        encoder_layer = EncoderLayer(d_model, nhead, 'linear')
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(
            2), 'the feature number of src and transformer must be equal'

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                src0, src1 = feat1, feat0
                feat0 = layer(feat0, src0, mask0, mask1)
                feat1 = layer(feat1, src1, mask1, mask0)
            else:
                raise KeyError
        return feat0, feat1


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, attention='linear'):
        super(DecoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.multihead_attn = MultiHeadAttention(d_model, nhead)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        # self.pre_norm_q = nn.LayerNorm(d_model)
        # self.pre_norm_kv = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_pos=None,
                m_pos=None):
        """
        Args:
            tgt (torch.Tensor): [N, L, C]
            memory (torch.Tensor): [N, S, C]
            tgt_mask (torch.Tensor): [N, L] (optional)
            memory_mask (torch.Tensor): [N, S] (optional)
        """
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, tgt_pos)
        tgt2 = self.self_attn(q, k, v=tgt2, q_mask=tgt_mask, kv_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            q=self.with_pos_embed(tgt2, tgt_pos),
            k=self.with_pos_embed(memory, m_pos),
            v=memory,
            q_mask=tgt_mask,
            kv_mask=memory_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.mlp(tgt2)
        tgt = tgt + tgt2

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_pos=None,
                m_pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_pos=tgt_pos,
                m_pos=m_pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class QueryTransformer(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 attention_mode='linear'):
        super(QueryTransformer, self).__init__()

        self.encoder_layer_names = ['self', 'cross'] * (num_layers)
        encoder_layer = EncoderLayer(d_model, nhead, attention=attention_mode)
        self.encoder = _get_clones(encoder_layer,
                                   len(self.encoder_layer_names))
        decoder_later = DecoderLayer(
            d_model,
            nhead,
            dropout=0.1,
        )
        self.decoder = TransformerDecoder(decoder_later, num_layers=2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        feat0,
        feat1,
        query_embed0,
        query_embed1,
        pos0,
        pos1,
        mask0=None,
        mask1=None,
    ):
        """
        Args:
            feat0:  [bs, c, h0, w0]
            feat1:
            query_embed0: [num_queries, c]
            query_embed1:
            pos0:
            pos1:
            mask0:
            mask1:
        Returns:
        """
        # pdb.set_trace()
        bs, c, h0, ww = feat0.shape
        feat0 = feat0.flatten(2).permute(0, 2, 1)
        feat1 = feat1.flatten(2).permute(0, 2, 1)
        if mask0 is not None:
            mask0 = mask0.flatten(1)
        if mask1 is not None:
            mask1 = mask1.flatten(1)
        pos0 = pos0.flatten(2).permute(0, 2, 1)
        pos1 = pos1.flatten(2).permute(0, 2, 1)
        query_embed0 = query_embed0.unsqueeze(0).repeat(bs, 1, 1)
        query_embed1 = query_embed1.unsqueeze(0).repeat(bs, 1, 1)

        for layer, name in zip(self.encoder, self.encoder_layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0, pos0, pos0)
                feat1 = layer(feat1, feat1, mask1, mask1, pos1, pos1)
            elif name == 'cross':
                src0, src1 = feat1, feat0
                feat0 = layer(feat0, src0, mask0, mask1, pos0, pos1)
                feat1 = layer(feat1, src1, mask1, mask0, pos1, pos0)
            else:
                raise KeyError

        # pdb.set_trace()
        tgt0 = torch.zeros_like(query_embed0)
        memory0 = feat0
        hs0 = self.decoder(
            tgt0,
            memory0,
            tgt_mask=None,
            memory_mask=mask0,
            tgt_pos=query_embed0,
            m_pos=pos0,
        )

        tgt1 = torch.zeros_like(query_embed1)
        memory1 = feat1
        hs1 = self.decoder(
            tgt1,
            memory1,
            tgt_mask=None,
            memory_mask=mask1,
            tgt_pos=query_embed1,
            m_pos=pos1,
        )
        # [bs, num_q, c]
        return hs0, hs1, memory0, memory1


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2,
                               1,
                               kernel_size,
                               padding=kernel_size // 2,
                               bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
