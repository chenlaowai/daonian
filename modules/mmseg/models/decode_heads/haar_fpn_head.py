# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import ModuleList, BaseModule
from mmseg.registry import MODELS
from mmseg.models.utils import Upsample, resize

from .basedecodehead_qffm import BaseDecodeHead_QFFM


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

@MODELS.register_module()
class Haar_FPN_Head(BaseDecodeHead_QFFM):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
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

        self.aux_conv = ConvBNReLU(self.in_channels[self.stage_num - 1], self.out_channels, kernel_size=1, stride=1)

    def forward(self, inputs):

        if self.training:
            input, outs_haar, outs_dhp = inputs[0], inputs[1], inputs[2]
        else:
            input = inputs

        out = self.scale_heads[0](input[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            out = out + resize(
                self.scale_heads[i](input[i]),
                size=out.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        out = self.cls_seg(out)

        outs_last = self.aux_conv(input[self.stage_num - 1])

        if self.training:
            return out, outs_haar, outs_dhp, outs_last
        else:
            return out
