_base_ = [
    '../_base_/datasets/datasets_iter.py',
    '../_base_/default_runtime_seg_iter.py', '../_base_/schedules/schedule_40k.py'
]

crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
depths=[2, 2, 6, 2]
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=768,
        channels=768,
        num_classes=7,
        num_layers=2,
        num_heads=6,
        embed_dims=768,
        dropout_ratio=0.0,
        # loss_decode=dict(
        #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # dict(type='CrossEntropyLoss', loss_name='lose_ce', use_sigmoid=False, loss_weight=1.0),
            # dict(type='LovaszLoss', loss_name='lose_lovasz', reduction='none', loss_weight=1.0),
            # dict(type='DiceLoss', loss_name='lose_dice', loss_weight=1.0)
                    ]
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=120000,
        by_epoch=False)
]
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'pos_block': dict(decay_mult=0.),
    #         'norm': dict(decay_mult=0.),
    #         'head': dict(lr_mult=10.),
    #         # 'bn': dict(decay_mult=0.0)
    #     },
    # ),
)

param_scheduler = [
    dict(type='LinearLR',
         start_factor=1e-6,
         by_epoch=False,
         begin=0,
         end=1500,
         ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=120000,
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=120000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, save_best='mIoU', interval=40000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))