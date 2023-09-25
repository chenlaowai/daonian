_base_ = [
    '../_base_/datasets/datasets_iter.py',
    '../_base_/default_runtime_seg_iter.py', '../_base_/schedules/schedule_40k.py'
]
num_classes = 2
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder_Plus',
    # pretrained='./model_data/mit_b1.pth',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='nextvit_small',
                  frozen_stages=-1,
                  norm_eval=False,
                  with_extra_norm=True,
                  norm_cfg=norm_cfg,
                  resume='../model_data/nextvit_small_in1k_224.pth',
                  ),
    decode_head=dict(
                     type='SegformerHead',
                     in_channels=[96, 256, 512, 1024],
                     in_index=[0, 1, 2, 3],
                     channels=512,
                     dropout_ratio=0.1,
                     num_classes=num_classes,
                     norm_cfg=norm_cfg,
                     align_corners=False,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                     ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            # 'bn': dict(decay_mult=0.0)
        },
    ),
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
        end=40000,
        by_epoch=False,
    )
]
# param_scheduler = [
#     dict(type='LinearLR',
#          start_factor=1e-6,
#          by_epoch=True,
#          begin=0,
#          end=2,
#          ),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=2,
#         end=12,
#         by_epoch=True,
#     )
# ]

train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader