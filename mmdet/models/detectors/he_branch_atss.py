# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .he_baseone import HEbaseOneDetector
from ..edge import *
import torch
from mmdet.core import bbox2result
import torch.nn.functional as F
import torch.nn as nn

@DETECTORS.register_module()
class HEBHATSS(HEbaseOneDetector):
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
        super(HEBHATSS, self).__init__(backbone,dehaze_model, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.dehaze_model is not None:
            device = img.device
            # methods = ['Roberts', 'Prewitt', 'Canny', 'Scharr', 'Sobel', 'Laplacian', 'LOG']
            methods = ['Scharr','Prewitt']   # # 将多种边缘算子得到的结果加到一起
            edge_out = edge_select(methods,img)
            # 把数据放在gpu上
            edge_out=edge_out.to(device)
            edge_imgs = self.dehaze_model(edge_out)  ##得到的结果经过分支模型

            # 将res18得到的结果通道数通过1×1卷积，改成与resnet50输出通道数一样
            # self.f = nn.ModuleList().to(device)
            # c=64
            # for i in range(len(x)):
            #     l_c =nn.Conv2d(c*2**i,c*2**(i+2),1,1,0)
            #     self.f.append(l_c)
            # edge =[]
            # for i in range(len(x)):
            #     edge.append(self.f[i](edge_imgs[i]))
            # edge_imgs=edge

        # 将经过分支的结果采样到与x相同的尺寸
        assert len(x)==len(edge_imgs) ,'len(x)!=len(edge_imgs)'
        # for i in range(len(x)):
        #     x_shape = x[i ].shape[2:]
        #     edge_imgs[i] = F.interpolate(edge_imgs[i], size=x_shape)
        # # # 把backbone的结果和提取边缘经过dehaze_model的结果之间相加
        y=[0.9*x[i]+0.1*edge_imgs[i] for i in range(len(x))]

        if self.with_neck:
            x = self.neck(y)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

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
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
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
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
