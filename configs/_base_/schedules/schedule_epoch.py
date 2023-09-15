# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]
# training schedule for 40k
#train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=50)
train_cfg = dict(type='EpochBasedTrainLoop_Flow', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop_Loss')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    #logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    logger=dict(type='LoggerHook_Plus', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    #checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000),
    checkpoint=dict(type='CheckpointHook_Plus', by_epoch=True, interval=9999, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

