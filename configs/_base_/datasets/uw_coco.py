# dataset settings
dataset_type = 'UWCocoDataset'
data_root = '/home/tju531/hwr/Datasets/uwdatasets/'
# img_norm_cfg = dict(
#     mean=[[123.675, 116.28, 103.53],128], std=[[58.395, 57.12, 57.375],57.12], to_rgb=True)
# mean[0]是img 的，mean[1]是edge_gt的

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='GetEdge',only_bbox=False),  ### only_bbox = Ture 是只求bbox中 目标的边缘，Fasle时求整张图片的边缘
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),# img_scale=(1000, 600)
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),   # gt_edge - > 如果不行 mean = [128,] std = [1]
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'edge']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/train.json',
        img_prefix=data_root + 'JPEGImages',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/test.json',
        img_prefix=data_root+'JPEGImages',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/val.json',
        img_prefix=data_root+'JPEGImages',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
