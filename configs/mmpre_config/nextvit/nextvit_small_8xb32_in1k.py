# model settings
_base_ = [
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]


norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='ImageClassifier',
    backbone=dict(type='NextViT',
                  stem_chs=[64, 32, 64],
                  depths=[3, 4, 10, 3],
                  out_indices=(3, ),
                  path_dropout=0.,
                  frozen_stages=-1,
                  norm_eval=False,
                  with_extra_norm=True,
                  ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))