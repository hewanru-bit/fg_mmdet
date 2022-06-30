# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .hebase import HEbaseDetector


@DETECTORS.register_module()
class HEFasterRCNN(HEbaseDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 dehaze_model,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(HEFasterRCNN, self).__init__(
            backbone=backbone,
            dehaze_model=dehaze_model,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
