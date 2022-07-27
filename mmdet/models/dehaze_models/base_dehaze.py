# Copyright (c) OpenMMLab. All rights reserved.
# Wang YuDong
from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule

from ..builder import build_loss


class BaseDehaze(BaseModule, metaclass=ABCMeta):
    """Base dehaze"""

    def __init__(self,
                 loss_dehaze=dict(type='L1Loss', loss_weight=1.0),
                 loss_perceptual=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if isinstance(loss_dehaze, dict):
            self.loss_dehaze = build_loss(loss_dehaze)
        elif isinstance(loss_dehaze, list):
            self.loss_dehaze = [
                build_loss(loss_cfg) for loss_cfg in loss_dehaze
            ]
        elif loss_dehaze is None:
            self.loss_dehaze = None
        else:
            raise KeyError
        if loss_perceptual is not None:
            self.loss_perseptual = build_loss(loss_perceptual)
        else:
            self.loss_perseptual = None

    @abstractmethod
    def loss(self, *args, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def forward(self, x):
        """Placeholder of forward function.

        Returns:
            Tensor: Prediction image.
        """
        pass

    @abstractmethod
    def get_results(self, *args, **kwargs):
        """Get results."""
        pass

    def forward_train(self,
                      img,
                      img_metas,
                      gt_img,
                      return_dehaze_img=False,
                      **kwargs):
        """
        Args:
            img (Tensor): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_img (Tensor): Ground truth images, has a shape
                (num_gts, H, W).

        Returns:
            losses: (dict): A dictionary of loss components.
        """
        dehaze_img = self(img)
        assert gt_img is not None
        assert dehaze_img.shape == img.shape
        losses = self.loss(
            *(dehaze_img, ), gt_img=gt_img, img_metas=img_metas)

        if return_dehaze_img:
            return dehaze_img, losses
        else:
            return losses

    def simple_test(self, x, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        outs = self(x)
        results_list = self.get_results(outs, img_metas, rescale)
        return results_list
