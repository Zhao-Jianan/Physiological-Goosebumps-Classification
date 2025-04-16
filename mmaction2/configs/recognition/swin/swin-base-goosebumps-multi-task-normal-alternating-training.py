_base_ = [
    '../../_base_/models/swin_tiny.py', '../../_base_/default_runtime.py'
]

model = dict(
    cls_head=dict(num_classes=2,in_channels=1024),
    backbone=dict(
        arch='base',
        drop_path_rate=0.3,
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_base_patch4_window7_224.pth'  # noqa: E501
    ))
load_from = 'checkpoint/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-182ec6cc.pth'


# dataset settings
dataset_type = 'MultiTaskVideoDataset'
data_root = 'data/Goosebumps_videos_multi_task'
data_root_val = 'data/Goosebumps_videos_multi_task'


file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=10, frame_interval=1, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=10,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=10,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]


ann_file_train_template = 'data/Goosebumps_videos_multi_task/{task}_normal_train_video.txt'
ann_file_val_template = 'data/Goosebumps_videos_multi_task/{task}_normal_val_video.txt'
ann_file_test_template = 'data/Goosebumps_videos_multi_task/{task}_normal_test_video.txt'

# Specify the tasks
tasks = ['Dom_arm', 'Dom_calf', 'Left_thigh', 'Right_thigh']
current_task_idx = 0

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train_template.format(task=tasks[current_task_idx]), 
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val_template.format(task=tasks[current_task_idx]),
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test_template.format(task=tasks[current_task_idx]),
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop', )
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05),
    accumulative_counts=4,
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=30)
]

log_processor = dict(
    type='LogProcessor',  
    window_size=20, 
    by_epoch=True)  
vis_backends = [  
    dict(type='LocalVisBackend')] 
visualizer = dict(  
    type='ActionVisualizer',  
    vis_backends=vis_backends)
log_level = 'INFO'  


default_hooks = dict(
    checkpoint=dict(interval=4, max_keep_ckpts=5), logger=dict(interval=100))
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision', 'mmit_mean_average_precision'])
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)

# Add custom hooks
custom_hooks = [
    dict(
        type='TaskSwitchHook',
        tasks=tasks,
        epochs_per_task=1,
        priority='HIGH')
]
