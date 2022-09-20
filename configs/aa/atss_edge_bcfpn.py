_base_ = [
    # '../_base_/datasets/coco_detection.py',
    # '../_base_/datasets/fgcoco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='CATSS',
    dehaze_model=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='BCFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        down_up=True,
        cat_feats=False,
        shape_level=0,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='CATSSHead',
        num_classes=5,  ###############
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_edge=dict(type='L1Loss', loss_weight=0.7)
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


dataset_type = 'FGCocoDataset'
data_root = '/home/tju531/hwr/Datasets/RESIDE/RTTS/'
img_norm_cfg = dict(
    mean=[[123.675, 116.28, 103.53],128], std=[[58.395, 57.12, 57.375],57.12], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='GetEdge',only_bbox=False),  ### only_bbox = Ture 是只求bbox中 目标的边缘，Fasle时求整张图片的边缘
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),# img_scale=(1000, 600)
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),   # gt_edge - > 如果不行 mean = [128,] std = [1]
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'edge']),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
        ann_file=data_root + 'rtts_coco/rtts_train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'rtts_coco/rtts_test.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'rtts_coco/rtts_test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)  ######  8,2,lr=0.01

interval=1
checkpoint_config = dict(interval=interval,max_keep_ckpts=1)
evaluation = dict(
    save_best='auto',
    interval=interval,
    metric='bbox')  # VOC类型的可选： 'mAP', 'recall'，COCO类型可选proposal，bbox
log_config = dict(interval=50) # 打印Log信息的间隔

runner = dict(type='EpochBasedRunner', max_epochs=12)
