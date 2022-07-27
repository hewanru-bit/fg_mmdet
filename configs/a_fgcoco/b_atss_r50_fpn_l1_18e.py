'''edge_loss不放在检测头中，放在fpn之后，head之前。img经过bk+fpn,fpn设置 start_level=0, num_outs=6, fpn输出[0]来与edge做l1 loss
 新建一个检测器类 B系列，loss在fpn后，head前'''
_base_ = [
    '../_base_/datasets/fgcoco_detection.py',
    # '../_base_/datasets/fgvoc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='BATSS',
    dehaze_model=dict(
        type='AODNet',   ###只是方便求loss，并不经过网络
        loss_dehaze=dict(type='L1Loss', loss_weight=0.1)
    ),
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
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=6), ############# start_level=0, num_outs=6
    bbox_head=dict(
        type='ATSSHead',
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
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
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
# optimizer
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)

interval=1
checkpoint_config = dict(interval=interval,max_keep_ckpts=1)
evaluation = dict(
    save_best='auto',
    interval=interval,
    metric='bbox')  #### VOC类型的可选： 'mAP', 'recall'，COCO类型可选proposal，bbox
log_config = dict(interval=100) ########打印Log信息的间隔

runner = dict(type='EpochBasedRunner', max_epochs=18)