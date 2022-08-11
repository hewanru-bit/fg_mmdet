# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from ..builder import HEADS, build_loss
from ..builder import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module()
class ARPNHead(AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        num_convs (int): Number of convolution layers in the head. Default 1.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 loss_edge=dict(
                     type='L1Loss',
                     loss_weight=1.0),
                 num_convs=1,
                 **kwargs):
        self.num_convs = num_convs
        super(ARPNHead, self).__init__(
            1, in_channels, init_cfg=init_cfg, **kwargs)

        self.loss_edge = build_loss(loss_edge)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4,
                                 1)

        # 在head中添加关注边缘的分支
        self.edge_convs = nn.ModuleList()
        for i in range(self.num_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.edge_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    inplace=False))
        # 将egde分支的结果输出为1通道
        self.edge_bh = nn.Conv2d(
            self.feat_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=1)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""

        edge_feat = x
        ### 添加edge分支
        for edge_conv in self.edge_convs:
            edge_feat = edge_conv(edge_feat)

        edge_imgs = self.edge_bh(edge_feat)

        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)

        return rpn_cls_score, rpn_bbox_pred, edge_imgs

    def loss(self,
             cls_scores,
             bbox_preds,
             edge_imgs,  ###############
             gt_bboxes,
             img_metas,
             edge,   ###############
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super(ARPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)

        ###########求edge loss #########################
        edge_target = self.edge_target(edge_imgs[0], edge)
        loss_edge = self.loss_edge(edge_imgs[0], edge_target)

        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'], loss_edge=loss_edge)

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_anchors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RPN head does not need this value.
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']

        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds,
                                       mlvl_valid_anchors, level_ids, cfg,
                                       img_shape)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
                           level_ids, cfg, img_shape, **kwargs):
        """bbox post-processing method.

        Do the nms operation for bboxes in same level.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            mlvl_valid_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_bboxes, 4).
            level_ids (list[Tensor]): Indexes from all scale levels of a
                single image, each item has shape (num_bboxes, ).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, `self.test_cfg` would be used.
            img_shape (tuple(int)): The shape of model's input image.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]


    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      edge=None,    ###########
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            gt_edge:b,1,h,w，图片求得的边缘,尺寸为原始img尺寸
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas,edge)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, edge)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    ####得到 边缘的gt
    def edge_target(self, edge_img, gt_edge):
        # 1.将gt_edge(b,c,h,w) 下采样到edge_imgs[0]的尺寸
        ####edge_img,gt_edge 都没有了bz通道
        w, h = edge_img.size()[2:]
        gt_edge = F.interpolate(gt_edge, size=(w, h))
        return gt_edge

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

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
                with shape (n,)
        """
        ###与原来的相比outs多了一个返回的结果，edge_imgs，但是测试时不使用，
        # self.get_bboxes中传参数，是原本的前三个，所以这里把edge_imgs去掉
        outs = self.forward(feats)[:-1]
        results_list = self.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale)
        return results_list