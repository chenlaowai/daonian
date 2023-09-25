_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/datasets_iter.py', '../_base_/default_runtime_seg_iter.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
num_classes = 7
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet18_v1c',
    # pretrained='../model_data/resnet18.pth',
    backbone=dict(depth=18,
                  init_cfg=dict(type='Pretrained', checkpoint='../model_data/resnet18.pth', prefix='backbone.')
                  ),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
        num_classes=num_classes,
    ),
    auxiliary_head=dict(in_channels=256, channels=64,
	                  num_classes=num_classes))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            # 'bn': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)

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
train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader