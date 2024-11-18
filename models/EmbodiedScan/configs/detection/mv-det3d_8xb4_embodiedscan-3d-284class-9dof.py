_base_ = ['../default_runtime.py']
n_points = 100000

backend_args = None
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/scannet/':
#         's3://openmmlab/datasets/detection3d/scannet_processed/',
#         'data/scannet/':
#         's3://openmmlab/datasets/detection3d/scannet_processed/'
#     }))

model = dict(
    type='SparseFeatureFusionSingleStage3DDetector',
    data_preprocessor=dict(type='Det3DDataPreprocessor',
                           mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375],
                           bgr_to_rgb=True,
                           pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        base_channels=16,  # to make it consistent with mink resnet
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    backbone_3d=dict(type='MinkResNet', in_channels=3, depth=34),
    use_xyz_feat=True,
    bbox_head=dict(type='FCAF3DHeadRotMat',
                   in_channels=(128, 256, 512, 1024),
                   out_channels=128,
                   voxel_size=.01,
                   pts_prune_threshold=100000,
                   pts_assign_threshold=27,
                   pts_center_threshold=18,
                   num_classes=284,
                   num_reg_outs=12,
                   center_loss=dict(type='mmdet.CrossEntropyLoss',
                                    use_sigmoid=True),
                   bbox_loss=dict(type='BBoxCDLoss',
                                  mode='l1',
                                  loss_weight=1.0,
                                  group='g8'),
                   cls_loss=dict(type='mmdet.FocalLoss'),
                   decouple_bbox_loss=True,
                   decouple_groups=4,
                   decouple_weights=[0.2, 0.2, 0.2, 0.4]),
    coord_type='DEPTH',
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

dataset_type = 'EmbodiedScanDataset'
data_root = 'data'
class_names = (
    'adhesive tape', 'air conditioner', 'alarm', 'album', 'arch', 'backpack',
    'bag', 'balcony', 'ball', 'banister', 'bar', 'barricade', 'baseboard',
    'basin', 'basket', 'bathtub', 'beam', 'beanbag', 'bed', 'bench', 'bicycle',
    'bidet', 'bin', 'blackboard', 'blanket', 'blinds', 'board', 'body loofah',
    'book', 'boots', 'bottle', 'bowl', 'box', 'bread', 'broom', 'brush',
    'bucket', 'cabinet', 'calendar', 'camera', 'can', 'candle', 'candlestick',
    'cap', 'car', 'carpet', 'cart', 'case', 'chair', 'chandelier', 'cleanser',
    'clock', 'clothes', 'clothes dryer', 'coat hanger', 'coffee maker', 'coil',
    'column', 'commode', 'computer', 'conducting wire', 'container', 'control',
    'copier', 'cosmetics', 'couch', 'counter', 'countertop', 'crate', 'crib',
    'cube', 'cup', 'curtain', 'cushion', 'decoration', 'desk', 'detergent',
    'device', 'dish rack', 'dishwasher', 'dispenser', 'divider', 'door',
    'door knob', 'doorframe', 'doorway', 'drawer', 'dress', 'dresser', 'drum',
    'duct', 'dumbbell', 'dustpan', 'dvd', 'eraser', 'excercise equipment',
    'fan', 'faucet', 'fence', 'file', 'fire extinguisher', 'fireplace',
    'flowerpot', 'flush', 'folder', 'food', 'footstool', 'frame', 'fruit',
    'furniture', 'garage door', 'garbage', 'glass', 'globe', 'glove',
    'grab bar', 'grass', 'guitar', 'hair dryer', 'hamper', 'handle', 'hanger',
    'hat', 'headboard', 'headphones', 'heater', 'helmets', 'holder', 'hook',
    'humidifier', 'ironware', 'jacket', 'jalousie', 'jar', 'kettle',
    'keyboard', 'kitchen island', 'kitchenware', 'knife', 'label', 'ladder',
    'lamp', 'laptop', 'ledge', 'letter', 'light', 'luggage', 'machine',
    'magazine', 'mailbox', 'map', 'mask', 'mat', 'mattress', 'menu',
    'microwave', 'mirror', 'molding', 'monitor', 'mop', 'mouse', 'napkins',
    'notebook', 'ottoman', 'oven', 'pack', 'package', 'pad', 'pan', 'panel',
    'paper', 'paper cutter', 'partition', 'pedestal', 'pen', 'person', 'piano',
    'picture', 'pillar', 'pillow', 'pipe', 'pitcher', 'plant', 'plate',
    'player', 'plug', 'plunger', 'pool', 'pool table', 'poster', 'pot',
    'price tag', 'printer', 'projector', 'purse', 'rack', 'radiator', 'radio',
    'rail', 'range hood', 'refrigerator', 'remote control', 'ridge', 'rod',
    'roll', 'roof', 'rope', 'sack', 'salt', 'scale', 'scissors', 'screen',
    'seasoning', 'shampoo', 'sheet', 'shelf', 'shirt', 'shoe', 'shovel',
    'shower', 'sign', 'sink', 'soap', 'soap dish', 'soap dispenser', 'socket',
    'speaker', 'sponge', 'spoon', 'stairs', 'stall', 'stand', 'stapler',
    'statue', 'steps', 'stick', 'stool', 'stopcock', 'stove', 'structure',
    'sunglasses', 'support', 'switch', 'table', 'tablet', 'teapot',
    'telephone', 'thermostat', 'tissue', 'tissue box', 'toaster', 'toilet',
    'toilet paper', 'toiletry', 'tool', 'toothbrush', 'toothpaste', 'towel',
    'toy', 'tray', 'treadmill', 'trophy', 'tube', 'tv', 'umbrella', 'urn',
    'utensil', 'vacuum cleaner', 'vanity', 'vase', 'vent', 'ventilation',
    'wardrobe', 'washbasin', 'washing machine', 'water cooler', 'water heater',
    'window', 'window frame', 'windowsill', 'wine', 'wire', 'wood', 'wrap')
head_labels = [
    48, 177, 82, 179, 37, 243, 28, 277, 32, 84, 215, 145, 182, 170, 22, 72, 30,
    141, 65, 257, 221, 225, 52, 75, 231, 158, 236, 156, 47, 74, 6, 18, 71, 242,
    217, 251, 66, 263, 5, 45, 14, 73, 278, 198, 24, 23, 196, 252, 19, 135, 26,
    229, 183, 200, 107, 272, 246, 269, 125, 59, 279, 15, 163, 258, 57, 195, 51,
    88, 97, 58, 102, 36, 137, 31, 80, 160, 155, 61, 238, 96, 190, 25, 219, 152,
    142, 201, 274, 249, 178, 192
]
common_labels = [
    189, 164, 101, 205, 273, 233, 131, 180, 86, 220, 67, 268, 224, 270, 53,
    203, 237, 226, 10, 133, 248, 41, 55, 16, 199, 134, 99, 185, 2, 20, 234,
    194, 253, 35, 174, 8, 223, 13, 91, 262, 230, 121, 49, 63, 119, 162, 79,
    168, 245, 267, 122, 104, 100, 1, 176, 280, 140, 209, 259, 143, 165, 147,
    117, 85, 105, 95, 109, 207, 68, 175, 106, 60, 4, 46, 171, 204, 111, 211,
    108, 120, 157, 222, 17, 264, 151, 98, 38, 261, 123, 78, 118, 127, 240, 124
]
tail_labels = [
    76, 149, 173, 250, 275, 255, 34, 77, 266, 283, 112, 115, 186, 136, 256, 40,
    254, 172, 9, 212, 213, 181, 154, 94, 191, 193, 3, 130, 146, 70, 128, 167,
    126, 81, 7, 11, 148, 228, 239, 247, 21, 42, 89, 153, 161, 244, 110, 0, 29,
    114, 132, 159, 218, 232, 260, 56, 92, 116, 282, 33, 113, 138, 12, 188, 44,
    150, 197, 271, 169, 206, 90, 235, 103, 281, 184, 208, 216, 202, 214, 241,
    129, 210, 276, 64, 27, 87, 139, 227, 187, 62, 43, 50, 69, 93, 144, 166,
    265, 54, 83, 39
]
metainfo = dict(classes=class_names,
                classes_split=(head_labels, common_labels, tail_labels),
                box_type_3d='euler-depth')

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='MultiViewPipeline',
         n_images=20,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA'),
             dict(type='PointSample', num_points=n_points // 10),
             dict(type='Resize', scale=(480, 480), keep_ratio=False)
         ]),
    dict(type='AggregateMultiViewPoints', coord_type='DEPTH'),
    dict(type='PointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_2d=False,  # only flip points
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.087266, 0.087266],
         scale_ratio_range=[.9, 1.1],
         translation_std=[.1, .1, .1],
         shift_height=False),
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='MultiViewPipeline',
         n_images=50,
         ordered=True,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA'),
             dict(type='PointSample', num_points=n_points // 10),
             dict(type='Resize', scale=(480, 480), keep_ratio=False)
         ]),
    dict(type='AggregateMultiViewPoints', coord_type='DEPTH'),
    dict(type='PointSample', num_points=n_points),
    dict(type='Pack3DDetInputs',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# TODO: to determine a reasonable batch size
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='RepeatDataset',
                 times=10,
                 dataset=dict(type=dataset_type,
                              data_root=data_root,
                              ann_file='embodiedscan_infos_train.pkl',
                              pipeline=train_pipeline,
                              test_mode=False,
                              filter_empty_gt=True,
                              box_type_3d='Euler-Depth',
                              metainfo=metainfo)))

val_dataloader = dict(batch_size=1,
                      num_workers=1,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   ann_file='embodiedscan_infos_val.pkl',
                                   pipeline=test_pipeline,
                                   test_mode=True,
                                   filter_empty_gt=True,
                                   box_type_3d='Euler-Depth',
                                   metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(type='IndoorDetMetric')
test_evaluator = val_evaluator

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW',
                                    lr=0.001,
                                    weight_decay=0.0001),
                     clip_grad=dict(max_norm=10, norm_type=2))

# learning rate
param_scheduler = dict(type='MultiStepLR',
                       begin=0,
                       end=12,
                       by_epoch=True,
                       milestones=[8, 11],
                       gamma=0.1)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4))
