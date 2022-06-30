# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdet.models.dehaze_models import AODNet
from mmdet.models.dehaze_models import CONVMNet

def test_anchor_head_loss():
    """Tests anchor head loss when truth is empty and non-empty."""
    # s = 256
    # img_metas = [{
    #     'img_shape': (s, s, 3),
    #     'scale_factor': 1,
    #     'pad_shape': (s, s, 3)
    # }]
    #
    # cfg = mmcv.Config(
    #     dict(
    #         in_channels=(1, 1, 2, 2, 4),
    #         base_channels=3,
    #         out_channels=(3, 3, 3, 3, 3),
    #         num_stages=5,
    #         strides=(1, 1, 1, 1, 1),
    #         kernel_size=(1, 3, 5, 7, 3),
    #         padding=(0, 1, 2, 3, 1),
    #         bias=(True, True, True, True, True),
    #         act_cfg=dict(type='ReLU'),
    #         plugins=None,
    #         pretrained=None,
    #         init_cfg=None,
    #         norm_eval=False,))
    # self = AODNet(**cfg)

    cfg = mmcv.Config(
        dict(
            in_channels=(1, 256, 256, 512, 1024),
            out_channels=(256, 256, 512, 1024, 2048),
            num_convm=5,  ####构造5个CONVM
            out_indices=(1, 2, 3, 4),  ###输出个数
            strides=(1, 1, 2, 2, 2),
            kernel_size=(1, 3, 3, 3, 3),
            padding=(1, 1, 1, 1, 1),
            bias=(True, True, True, True, True),
            act_cfg=dict(type='ReLU'),
            plugins=None,
            pretrained=None,
            init_cfg=None,
            norm_eval=False,
            init_weights=True))
    self = CONVMNet(**cfg)

    x = torch.rand(2,1,256,256)
    cls_scores = self.forward(x)
    print(cls_scores.shape)

