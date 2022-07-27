# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdet.models.dehaze_models import AODNet
from mmdet.models.dehaze_models import CONVMNet
from mmdet.models.necks import ACFPN1

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
            in_channels=[8, 16, 32, 64],
            out_channels=16,
            num_outs=4,
            shape_level=[0, 1, 2, 3]
        ))
    self = ACFPN1(**cfg)

    x = torch.rand(1,1,40,60)
    cls_scores = self.forward(x)
    print(len(cls_scores))

