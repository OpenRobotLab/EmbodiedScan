
# edit it
load_from = '/path/to/mv-3ddet.pth'
backend_args = None
custom_hooks = [
    dict(after_iter=True, type='EmptyCacheHook'),
]
data_root = 'data'
dataset_type = 'PointCloud3DGroundingDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'embodiedscan'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl', port=22873),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
launcher = 'slurm'

log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.0005
metainfo = dict(classes='all')
model = dict(
    backbone_3d=dict(depth=34, in_channels=6, type='MinkResNet'),
    bbox_head=dict(
        contrastive_cfg=dict(bias=True, log_scale='auto', max_text_len=256),
        decouple_bbox_loss=True,
        decouple_groups=4,
        decouple_weights=[
            0.2,
            0.2,
            0.2,
            0.4,
        ],
        loss_bbox=dict(
            group='g8', loss_weight=1.0, mode='l1', type='BBoxCDLoss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        num_classes=256,
        share_pred_layer=True,
        sync_cls_avg_factor=True,
        type='GroundingHead'),
    coord_type='DEPTH',
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='Det3DDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            cross_attn_text_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    neck_3d=dict(
        in_channels=[
            64,
            128,
            256,
            512,
        ],
        num_classes=1,
        out_channels=256,
        pts_prune_threshold=1000,
        type='MinkNeck',
        voxel_size=0.01),
    num_queries=256,
    test_cfg=None,
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=1.0),
                dict(type='BBox3DL1Cost', weight=2.0),
                dict(type='IoU3DCost', weight=2.0),
            ],
            type='HungarianAssigner3D')),
    type='SparseFeatureFusion3DGrounderMod',
    use_xyz_feat=True,
    voxel_size=0.01)
n_points = 100000
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.0005, type='AdamW', weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys=dict(
            decoder=dict(decay_mult=1.0, lr_mult=0.1),
            text_encoder=dict(lr_mult=0.0))),
    type='OptimWrapper')
param_scheduler = dict(
    begin=0,
    by_epoch=True,
    end=12,
    gamma=0.1,
    milestones=[
        8,
        11,
    ],
    type='MultiStepLR')
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=24,
    dataset=dict(
        ann_file='embodiedscan_infos_val.pkl',
        box_type_3d='Euler-Depth',
        data_root='data',
        filter_empty_gt=True,
        metainfo=dict(classes='all'),
        pipeline=[
            dict(type='LoadAnnotations3D'),
            dict(type='DefaultPipeline'),
            dict(num_points=100000, type='PointSample'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        tokens_positive_rebuild=True,
        type='MMScanPointCloud3DGroundingDataset',
        vg_file=
        ''),
    drop_last=False,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='GroundingMetricMod')
test_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='DefaultPipeline'),
    dict(num_points=100000, type='PointSample'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=3)
train_dataloader = dict(
    batch_size=24,
    dataset=dict(
        dataset=dict(
            ann_file='embodiedscan_infos_train.pkl',
            box_type_3d='Euler-Depth',
            data_root='data',
            filter_empty_gt=True,
            metainfo=dict(classes='all'),
            pipeline=[
                dict(type='LoadAnnotations3D'),
                dict(type='DefaultPipeline'),
                dict(num_points=100000, type='PointSample'),
                dict(
                    rot_range=[
                        -0.087266,
                        0.087266,
                    ],
                    scale_ratio_range=[
                        0.9,
                        1.1,
                    ],
                    shift_height=False,
                    translation_std=[
                        0.1,
                        0.1,
                        0.1,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    keys=[
                        'points',
                        'gt_bboxes_3d',
                        'gt_labels_3d',
                    ],
                    type='Pack3DDetInputs'),
            ],
            test_mode=False,
            tokens_positive_rebuild=True,
            type='MMScanPointCloud3DGroundingDataset',
            vg_file=
            ''
        ),
        times=1,
        type='RepeatDataset'),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='DefaultPipeline'),
    dict(num_points=100000, type='PointSample'),
    dict(
        rot_range=[
            -0.087266,
            0.087266,
        ],
        scale_ratio_range=[
            0.9,
            1.1,
        ],
        shift_height=False,
        translation_std=[
            0.1,
            0.1,
            0.1,
        ],
        type='GlobalRotScaleTrans'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=24,
    dataset=dict(
        ann_file='embodiedscan_infos_val.pkl',
        box_type_3d='Euler-Depth',
        data_root='data',
        filter_empty_gt=True,
        metainfo=dict(classes='all'),
        pipeline=[
            dict(type='LoadAnnotations3D'),
            dict(type='DefaultPipeline'),
            dict(num_points=100000, type='PointSample'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        tokens_positive_rebuild=True,
        type='MMScanPointCloud3DGroundingDataset',
        vg_file=
        ''),
    drop_last=False,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='GroundingMetricMod')
work_dir = '/mnt/petrelfs/lvruiyuan/repos/EmbodiedScan/work_dirs/pcd-mmscan-grounding-20Per-100queries-load'
