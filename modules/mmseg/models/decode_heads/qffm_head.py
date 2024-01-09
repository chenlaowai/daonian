# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import ModuleList, BaseModule
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)

from mmseg.registry import MODELS
from mmseg.models.utils import Upsample, resize

from .basedecodehead_qffm import BaseDecodeHead_QFFM


# SE注意力机制
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBNReLU(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, conv_cfg=None, norm_cfg=dict(type='BN2d'), init_cfg=None):
        super().__init__(init_cfg)
        self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False)
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm_name, norm)
        self.act = nn.ReLU(inplace=True)

    @property
    def norm(self):
        """shaboyi"""
        return getattr(self, self.norm_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x

class QFFMBlock(BaseModule):
    """

    Args:

    """

    def __init__(self,
                 channel,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.QFFM_W = nn.Parameter(torch.zero((channel, 1)))
        # self.QFFM_W = nn.Embedding(channel, 1)

        self.se = SELayer(channel, reduction=16)

    def init_weights(self):
        trunc_normal_(self.QFFM_W, std=0.02)

    def forward(self, f_h, f_l):
        b_h, c_h, h_h, w_h = f_h.shape
        b_l, c_l, h_l, w_l = f_l.shape

        # self.QFFM_W_weight = self.QFFM_W.weight
        f_h = f_h.permute(0, 2, 3, 1).contiguous()
        U = f_h @ self.QFFM_W
        U = U.permute(0, 3, 1, 2).contiguous().view(b_h, -1, h_l * w_l)

        V = f_l.permute(0, 2, 3, 1).contiguous().view(b_l, h_l * w_l, c_l)

        UV = U @ V
        UV = UV.permute(0, 2, 1).contiguous().view(b_l, c_l, 1, 1)

        UV = self.se(UV)

        out = UV.expand(-1, -1, h_l, w_l)

        return out

class QFFMLayer(BaseModule):
    """

    Args:

    """

    def __init__(self,
                 high_channel,
                 low_channel,
                 align_corners=False,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.align_corners = align_corners
        # self.up_channel = ConvBNReLU(high_channel, low_channel, kernel_size=1, stride=1)
        self.qffmblock_hh = QFFMBlock(low_channel)
        self.qffmblock_ll = QFFMBlock(low_channel)
        self.qffmblock_hl = QFFMBlock(low_channel)

    def forward(self, out_h, out_l):
        # out_h = resize(
        #     input=out_h,
        #     size=out_l.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        #
        # out_h = self.up_channel(out_h)

        qffmblock_hh = self.qffmblock_hh(out_h, out_h)
        qffmblock_ll = self.qffmblock_ll(out_l, out_l)
        qffmblock_hl = self.qffmblock_hl(out_h, out_l)

        out = qffmblock_hh + qffmblock_ll + qffmblock_hl + out_h + out_l

        return out

@MODELS.register_module()
class QFFMHead(BaseDecodeHead_QFFM):
    """

    Args:

    """

    def __init__(self, feature_strides,
                 init_std=0.02,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.init_std = init_std
        self.qffmlayers = ModuleList()
        for i in range(self.stage_num - 1):
            qffmlayer = QFFMLayer(self.in_channels[self.stage_num - 1 - i], self.in_channels[self.stage_num - 2 - i])
            self.qffmlayers.append(qffmlayer)

        self.aux_conv = ConvBNReLU(self.in_channels[self.stage_num - 1], self.out_channels, kernel_size=1, stride=1)

        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            if i == 0:
                out_channel = self.channels
            else:
                out_channel = self.in_channels[i - 1]
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else out_channel,
                        out_channel,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)


    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        if self.training:
            input, outs_haar, outs_dhp = inputs[0], inputs[1], inputs[2]
        else:
            input = inputs

        high_feature = input[self.stage_num - 1]
        for i in range(self.stage_num - 1):
            high_feature = self.scale_heads[self.stage_num - 1 - i](high_feature)
            low_feature = input[self.stage_num - 2 - i]
            high_feature = resize(
                high_feature,
                size=low_feature.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            # high_feature = self.qffmlayers[i](high_feature, low_feature)
            high_feature = high_feature + low_feature

        out = self.cls_seg(high_feature)

        outs_last = self.aux_conv(input[self.stage_num - 1])

        if self.training:
            return out, outs_haar, outs_dhp, outs_last
        else:
            return out
