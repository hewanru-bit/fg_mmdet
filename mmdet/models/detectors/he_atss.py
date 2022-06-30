# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .he_baseone import HEbaseOneDetector


@DETECTORS.register_module()
class HEATSS(HEbaseOneDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 dehaze_model,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(HEATSS, self).__init__(backbone,dehaze_model, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
