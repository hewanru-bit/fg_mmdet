
_base_ = [
    # '../_base_/datasets/fgcoco_detection.py',
    # '../_base_/datasets/fgvoc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=5,  #######################
        in_channels=256,
        stacked_convs=4,
        center_sampling=True,
        center_sample_radius=1.5,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))


dataset_type = 'FGCocoDataset'
data_root = '/home/tju531/hwr/Datasets/RESIDE/RTTS/'
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
optimizer = dict(
    lr=0.0025, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))  ################## 原始lr=0.01
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])

checkpoint_config = dict(interval=1,max_keep_ckpts=1)
evaluation = dict(
    save_best='auto',
    interval=1,
    metric='bbox')  # VOC类型的可选： 'mAP', 'recall'，COCO类型可选proposal，bboX

log_config = dict(interval=50) # 打印Log信息的间隔
runner = dict(type='EpochBasedRunner', max_epochs=12)

