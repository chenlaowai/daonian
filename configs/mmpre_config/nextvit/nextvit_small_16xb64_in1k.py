# model settings
_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]


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
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)

optim_wrapper = dict(clip_grad=dict(max_norm=5.0))