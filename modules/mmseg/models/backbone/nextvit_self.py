# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from einops import rearrange
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer, ConvModule
from mmengine.model import BaseModule
from mmcv.cnn.bricks import DropPath
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS


NORM_EPS = 1e-5

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class PatchEmbed(BaseModule):
    def __init__(self, in_channels, out_channels, stride=1, conv_cfg=None, norm_cfg=dict(type='BN2d'), init_cfg=None):
        super().__init__(init_cfg)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    @property
    def norm(self):
        """shaboyi"""
        return getattr(self, self.norm_name)

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))

class ConvBNReLU(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, conv_cfg=None, norm_cfg=dict(type='BN2d'), init_cfg=None):
        super().__init__(init_cfg)
        self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, groups=groups, bias=False)
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

class MHCA(BaseModule):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, out_channels, head_dim, conv_cfg=None, norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__(init_cfg)
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm_name, norm)
        self.group_conv3x3 = build_conv_layer(conv_cfg, out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels // head_dim, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.projection = build_conv_layer(conv_cfg, out_channels, out_channels, kernel_size=1, bias=False)

    @property
    def norm(self):
        """shaboyi"""
        return getattr(self, self.norm_name)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)

        return out

class Mlp(BaseModule):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True, conv_cfg=None, init_cfg=None):
        super().__init__(init_cfg)
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = build_conv_layer(conv_cfg, in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = build_conv_layer(conv_cfg, hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)

        return x

class E_MHSA(BaseModule):
    """
    Efficient Multi-Head Self Attention
    """
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0., sr_ratio=1, norm_cfg=dict(type='BN1d'), init_cfg=None):
        super().__init__(init_cfg)
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm_name, norm = build_norm_layer(norm_cfg, dim)
            self.add_module(self.norm_name, norm)

    @property
    def norm(self):
        """shaboyi"""
        return getattr(self, self.norm_name)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if True:
            # if not torch.onnx.is_in_onnx_export()
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class NCB(BaseModule):
    """
    Next Convolution Block
    """
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0., drop=0, head_dim=32, mlp_ratio=3, norm_cfg=dict(type='BN2d'), init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm_name, norm)
        assert out_channels % head_dim == 0
        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.attention_path_dropout = DropPath(path_dropout)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        self.mlp_path_dropout = DropPath(path_dropout)

    @property
    def norm(self):
        """shaboyi"""
        return getattr(self, self.norm_name)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x))
        if True:
        # if not torch.onnx.is_in_onnx_export():
            out = self.norm(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x

class NTB(BaseModule):
    """
    Next Transformer Block
    """
    def __init__(self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1, mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0, norm_cfg=dict(type='BN2d'), init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio

        self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.mhsa_out_channels, postfix=1)
        self.add_module(self.norm_name, norm1)
        self.e_mhsa = E_MHSA(self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio, attn_drop=attn_drop, proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2_name, norm2 = build_norm_layer(norm_cfg, out_channels, postfix=2)
        self.add_module(self.norm_name, norm2)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = DropPath(path_dropout)

    @property
    def norm1(self):
        """shaboyi"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """shaboyi"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        if True:
        # if not torch.onnx.is_in_onnx_export():
            out = self.norm1(x)
        else:
            out = x
        out = rearrange(out, "b c h w -> b (h w) c")  # b n c
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        x = x + rearrange(out, "b (h w) c -> b c h w", h=H)

        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out))
        x = torch.cat([x, out], dim=1)

        if True:
        # if not torch.onnx.is_in_onnx_export():
            out = self.norm2(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x

@MODELS.register_module()
class NextViT(BaseModule):
    def __init__(self,
                 stem_chs,
                 depths,
                 path_dropout,
                 attn_drop=0,
                 drop=0,
                 strides=[1, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 head_dim=32,
                 mix_block_ratio=0.75,
                 use_checkpoint=False,
                 with_extra_norm=True,
                 frozen_stages=-1,
                 norm_eval=False,
                 norm_cfg=dict(type='BN2d'),
                 init_cfg=None
                 ):
        super().__init__(init_cfg)
        self.use_checkpoint = use_checkpoint
        self.frozen_stages = frozen_stages
        self.with_extra_norm = with_extra_norm
        self.norm_eval = norm_eval
        self.stage_out_channels = [[96] * (depths[0]),
                                   [192] * (depths[1] - 1) + [256],
                                   [384, 384, 384, 384, 512] * (depths[2] // 5),
                                   [768] * (depths[3] - 1) + [1024]]

        # Next Hybrid Strategy
        self.stage_block_types = [[NCB] * depths[0],
                                  [NCB] * (depths[1] - 1) + [NTB],
                                  [NCB, NCB, NCB, NCB, NTB] * (depths[2] // 5),
                                  [NCB] * (depths[3] - 1) + [NTB]]

        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is NCB:
                    layer = NCB(input_channel, output_channel, stride=stride, path_dropout=dpr[idx + block_id],
                                drop=drop, head_dim=head_dim)
                    features.append(layer)
                elif block_type is NTB:
                    layer = NTB(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                                sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio,
                                attn_drop=attn_drop, drop=drop)
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.extra_norm_list = None
        if with_extra_norm:
            self.extra_norm_list = []
            for stage_id in range(len(self.stage_out_channels)):
                self.extra_norm_list.append(nn.BatchNorm2d(
                    self.stage_out_channels[stage_id][-1], eps=NORM_EPS))
            self.extra_norm_list = nn.Sequential(*self.extra_norm_list)

        self.norm_name, norm = build_norm_layer(norm_cfg, output_channel)
        self.add_module(self.norm_name, norm)
        #
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.proj_head = nn.Sequential(
        #     nn.Linear(output_channel, num_classes),
        # )

        self.stage_out_idx = [sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]
        # if norm_cfg is not None:  # 用于分布式训练
        #     self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self._freeze_stages()

    @property
    def norm(self):
        """shaboyi"""
        return getattr(self, self.norm_name)

    def _freeze_stages(self):
        if self.frozen_stages > 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
            for idx, layer in enumerate(self.features):
                if idx <= self.stage_out_idx[self.frozen_stages - 1]:
                    layer.eval()
                    for param in layer.parameters():
                        param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(NextViT, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outputs = list()
        x = self.stem(x)
        stage_id = 0
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == self.stage_out_idx[stage_id]:
                if self.with_extra_norm:
                    if stage_id < 3:
                        x = self.extra_norm_list[stage_id](x)
                    else:
                        x = self.norm(x)
                outputs.append(x)
                stage_id += 1

        return outputs