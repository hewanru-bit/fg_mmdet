# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .he_baseone import HEbaseOneDetector
from ..edge import *
from mmdet.core import bbox2result
'''B系列，检测头不动，img过bk,fpn之后求一个边缘loss，dehaze_model只用1×1卷积，改变通道数和求loss'''
@DETECTORS.register_module()
class BATSS(HEbaseOneDetector):
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
        super(BATSS, self).__init__(backbone, dehaze_model, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""

        x = self.backbone(img)

        if self.with_neck:
            x = self.neck(x)
        if self.dehaze_model is not None:
            device = img.device
            # methods = ['Roberts', 'Prewitt', 'Canny', 'Scharr', 'Sobel', 'Laplacian', 'LOG']
            methods = ['Scharr']
            edge_out = edge_select(methods, img)
            # 把数据放在gpu上
            gt_edge = edge_out.to(device)
            return x, gt_edge

        return x


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(HEbaseOneDetector, self).forward_train(img, img_metas)
        x, gt_edge = self.extract_feat(img)
        # 取fpn的输出最底层，在dehaze_model（AODNet）中只是过一个1*1卷积，来匹配尺寸
        img_edge = x[0]
        losses = dict()
        if self.dehaze_model is not None:
            # 修改AODNet，不经过网络就是过一个1×1卷积和loss
            branch_loss = self.dehaze_model.forward_train(
                img=img_edge, img_metas=img_metas, gt_img=gt_edge)

            losses.update(branch_loss)

        main_losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                   gt_labels, gt_bboxes_ignore)
        losses.update(main_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat,gt_edge = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results