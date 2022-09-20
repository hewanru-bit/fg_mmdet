# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32
from .anchor_head import AnchorHead
from ..builder import HEADS, build_loss
import torch
import warnings
import numpy as np
from mmdet.core import (anchor_inside_flags, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, unmap)
import torch.nn.functional as F
from mmdet.utils import selcet_bboxes

'''在原始的RetinaHead 的基础上添加了 edge 分支'''
@HEADS.register_module()
class ARetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 loss_edge=dict(
                     type='L1Loss',
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        super(ARetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

        self.loss_edge = build_loss(loss_edge)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        # 在head中添加关注边缘的分支
        self.edge_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.edge_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

        ####将egde分支的结果输出为1通道
        self.edge_bh = nn.Conv2d(
            self.feat_channels,
            1,
            1,
            padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x

        edge_feat = x
        ### 添加edge分支
        for edge_conv in self.edge_convs:
            edge_feat = edge_conv(edge_feat)

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        edge_imgs = self.edge_bh(edge_feat)

        return cls_score, bbox_pred, edge_imgs

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      edge=None,
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
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas,edge)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds','edge_imgs'))
    def loss(self,
             cls_scores,
             bbox_preds,
             edge_imgs,  ############
             gt_bboxes,
             gt_labels,
             img_metas,
             edge,  ########
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        ###########求edge loss #########################
        loss_edge = self.bbox_edge_loss(edge_imgs, edge, bbox_preds, anchor_list,
                                        labels_list, gt_bboxes)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_edge=loss_edge)


    def bbox_edge_loss(self,edge_imgs,gt_edge, bbox_pred,
                       anchor, labels, gt_bboxes,getpreedge=0, digmed=True, use_prebbox=True):
        # getpreedge=0 表示直接使用网络输出的edge_imgs[0]进行操作
        # getpreedge=1 表示将网络输出的后四层，采样到edge_imgs[0]的尺寸，相加作为edge_img进行后续操作
        # 重新设置参数，digmed=Ture表示使用扣除，为Fasle表示不扣除
        # use_prebbox=Ture 表示使用gt——bboxesd 点进行扣除,use_prebbox=False表示用预测的点扣除

        # 1.按照getpreedge的方式得到edge_img
        if getpreedge:
            tems=0
            w, h = edge_imgs[0].size()[2:]
            for i in range(len(edge_imgs)):
                tem = F.interpolate(edge_imgs[i], size=(w, h))
                tems=tems+tem
            edge_img = tems
        else:
            edge_img = edge_imgs[0]

        # 2.按照digmed 是否进行扣除和扣除方式进一布得到edge_preds
        if digmed:
            # 必须edge_img 采样到gt——edge尺寸
            edge_size = self.edge_target(edge_img, gt_edge, sm=0)
            if use_prebbox:
                # use_gtbbox=False表示用预测的点扣除
                # 输入的anchor是bs个list 每个又包含5层list， bbox_pred是先5层，每层有bz个，
                # 都需要转成bz个list一层，即一张图片所有的anchor 和bboxespred为一个list
                concat_anchor=[]
                for i in range(len(anchor)):
                    concat_anchor.append(torch.cat(anchor[i]))

                # 先把bboxes_pred转成anchor一样的存储方式
                bboxes_pred_list=[]
                for i in range(len(anchor)):
                    sing_bbox_pred=[]
                    for j in range(len(bbox_pred)):
                        sing_bbox_pred.append(bbox_pred[j][i].permute(1, 2, 0).reshape(-1,4))
                    bboxes_pred_list.append(sing_bbox_pred)
                concat_bboxes_pred = []
                for i in range(len(bboxes_pred_list)):
                    concat_bboxes_pred.append(torch.cat(bboxes_pred_list[i]))

                # labels也要转化
                labels_list = []
                for i in range(len(anchor)):
                    sing_labels = []
                    for j in range(len(labels)):
                        sing_labels.append(labels[j][i])
                    labels_list.append(sing_labels)
                concat_labels = []
                for i in range(len(labels_list)):
                    concat_labels.append(torch.cat(labels_list[i]))

                # 单张图片处理
                edge_digs=[]
                for i in range(len(concat_labels)):
                    bg_class_ind = self.num_classes
                    pos_inds = ((concat_labels[i] >= 0)
                                & (concat_labels[i] < bg_class_ind)).nonzero().squeeze(1)
                    if len(pos_inds) > 0:
                        pos_bbox_pred = concat_bboxes_pred[i][pos_inds]
                        pos_anchors = concat_anchor[i][pos_inds]
                        # pos_anchors 是正常坐标，但 pos_bbox_pred是偏移量，先解码
                        use_bbox=self.bbox_coder.decode(pos_anchors, pos_bbox_pred)
                        edge_dig = selcet_bboxes(edge_size[i], use_bbox)
                        edge_digs.append(edge_dig)
                edge_pred = torch.stack(edge_digs,dim=0)
            else:
                edge_digs = []
                for i in range(len(gt_bboxes)):
                    edge_dig = selcet_bboxes(edge_size[i], gt_bboxes[i])
                    edge_digs.append(edge_dig)
                edge_pred = torch.stack(edge_digs, dim=0)
            loss_edge = self.loss_edge(edge_pred, gt_edge)
        else:
            # 不扣除bbox,sm=0,edge_img采样到gt——edge大小返回edge_size
            edge_size = self.edge_target(edge_img, gt_edge, sm=0)
            loss_edge = self.loss_edge(edge_size, gt_edge)

            # gt_edge才样至edge_img，返回edge_target
            # edge_target = self.edge_target(edge_img, gt_edge, sm=1)
            # loss_edge = self.loss_edge(edge_img, edge_target)

        return loss_edge

    def edge_target(self,edge_img, gt_edge, sm=1):
        if sm ==1:
            # 1.将gt_edge(b,c,h,w) 下采样到edge_imgs[0]的尺寸
            ####edge_img,gt_edge 都没有了bz通道
            w,h = edge_img.size()[2:]
            gt_edge = F.interpolate(gt_edge, size=(w,h))
            return gt_edge
        else:
            # 2.将edge_img上采样到gt_edge的尺寸
            w, h = gt_edge.size()[2:]
            edge_pre = F.interpolate(edge_img, size=(w, h))
            return edge_pre

    def simple_test_bboxes(self,feats, img_metas, rescale=False):
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