_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/defect_datasets_labelygd.py',
    '../_base_/default_runtime_seg_epoch.py', '../_base_/schedules/schedule_epoch.py'
]
num_classes = 3
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    pretrained='../model_data/mit_b1.pth',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[2, 2, 2, 2],
        #init_cfg=dict(type='Pretrained', checkpoint='model_data/mit_b1.pth')
    ),
    decode_head=dict(in_channels=[64, 128, 320, 512],
                     num_classes=num_classes,),
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        },
    ),
)

# param_scheduler = [
#     dict(type='LinearLR',
#          start_factor=1e-6,
#          by_epoch=False,
#          begin=0,
#          end=1500,
#          ),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=10000,
#         by_epoch=False,
#     )
# ]
param_scheduler = [
    dict(type='LinearLR',
         start_factor=1e-6,
         by_epoch=True,
         begin=0,
         end=2,
         ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=2,
        end=12,
        by_epoch=True,
    )
]

train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader