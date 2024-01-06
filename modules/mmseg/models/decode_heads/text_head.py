# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import ModuleList, BaseModule
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmseg.models.utils import resize

from mmseg.models.backbones.vit import TransformerEncoderLayer
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class QFFMBlock(BaseModule):
    """

    Args:

    """

    def __init__(self,
                 high_channel=768,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.QFFM_W = nn.Parameter(torch.randn((high_channel, 1)))
        # self.QFFM_W = nn.Embedding(torch.randn((high_channel, 1)))


    def init_weights(self):
        trunc_normal_(self.QFFM_W, std=0.02)

    def forward(self, f_h, f_l):
        b_h, h_h, w_h, c_h = f_h.shape
        b_l, h_l, w_l, c_l = f_l.shape

        if h_h < h_l:
            f_h = resize(
                input=f_h,
                size=f_l.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        U = f_h @ self.QFFM_W
        U = U.view(b_h, -1, h_l * w_l)

        V = f_l.permute(0, 2, 3, 1).contiguous().view(b_l, h_l * w_l, c_l)

        UV = U @ V
        UV = UV.view(b_l, 1, c_l, -1)


        out = UV.view(b_l, 1, 1, c_l).expand(1, h_l, w_l, 1).view(b_l, c_l, h_l, w_l)

        return out

class QFFMLayer(BaseModule):
    """

    Args:

    """

    def __init__(self,
                 low_channel=384,
                 high_channel=768,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.qffmblock_hh = QFFMBlock(high_channel)
        self.qffmblock_ll = QFFMBlock(low_channel)
        self.qffmblock_hl = QFFMBlock(high_channel)

    def forward(self, out_h, out_l):
        qffmblock_hh = self.qffmblock_hh(out_h, out_h)
        qffmblock_ll = self.qffmblock_ll(out_l, out_l)
        qffmblock_hl = self.qffmblock_ll(out_h, out_l)

        out_h_upsample = resize(
            input=out_h,
            size=out_l.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        out = qffmblock_hh + qffmblock_ll + qffmblock_hl + out_h_upsample + out_l

        return out

@MODELS.register_module()
class TextHead(BaseDecodeHead):
    """Segmenter: Transformer for Semantic Segmentation.

    This head is the implementation of
    `Segmenter: <https://arxiv.org/abs/2105.05633>`_.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        in_channels (int): The number of channels of input image.
        num_layers (int): The depth of transformer.
        num_heads (int): The number of attention heads.
        embed_dims (int): The number of embedding dimension.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        init_std (float): The value of std in weight initialization.
            Default: 0.02.
    """

    def __init__(
            self,
            in_channels=768,
            num_layers=2,
            num_heads=6,
            embed_dims=768,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            num_fcs=2,
            qkv_bias=True,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            init_std=0.02,
            **kwargs,
    ):
        super().__init__(in_channels=in_channels, **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True,
                ))

        self.dec_proj = nn.Linear(in_channels, embed_dims)

        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_classes, embed_dims))
        self.patch_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.classes_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        self.decoder_norm = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)[1]
        self.mask_norm = build_norm_layer(
            norm_cfg, self.num_classes, postfix=2)[1]

        self.init_std = init_std

        delattr(self, 'conv_seg')

        self.q = QFFMLayer()

    def init_weights(self):
        trunc_normal_(self.cls_emb, std=self.init_std)
        trunc_normal_init(self.patch_proj, std=self.init_std)
        trunc_normal_init(self.classes_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        x = self.dec_proj(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for layer in self.layers:
            x = layer(x)
        x = self.decoder_norm(x)

        patches = self.patch_proj(x[:, :-self.num_classes])
        cls_seg_feat = self.classes_proj(x[:, -self.num_classes:])

        patches = F.normalize(patches, dim=2, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)

        return masks
