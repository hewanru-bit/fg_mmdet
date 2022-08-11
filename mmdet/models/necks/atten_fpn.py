# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from ..builder import NECKS
from mmdet.models.attention import CBAMBlock, SEAttention, SKAttention


@NECKS.register_module()
class ATTENFPN(BaseModule):
    """
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 down_up=False,  ####
                 cat_feats=False,
                 shape_level=2,  # 平均池化的尺寸与这一层相同
                 pooling_type='AVG',
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ATTENFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.down_up = down_up
        self.cat_feats =cat_feats
        self.shape_level = shape_level

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpl_convs = nn.ModuleList()
        self.atten_layers = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)
            self.lateral_convs.append(l_conv)


        for j in range(self.num_outs):
            fpl_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            atten_layer = SEAttention(              #########################
                channel = out_channels, out_channel=out_channels, reduction= 16
            )
            self.fpl_convs.append(fpl_conv)
            self.atten_layers.append(atten_layer)

        if self.cat_feats:
            self.fp1_convs = ConvModule(
                out_channels * self.num_outs,
                out_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.atten = SEAttention(
                channel = out_channels, out_channel=out_channels, reduction=16
            )

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.adaptive_avg_pool2d  ##自适应全局平均池

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        laterals = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # 增加额外的层
        if self.num_outs > len(laterals):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    laterals.append(F.max_pool2d(laterals[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = laterals[-1]
                else:
                    raise NotImplementedError
                laterals.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        laterals.append(self.fpn_convs[i](F.relu(laterals[-1])))
                    else:
                        laterals.append(self.fpn_convs[i](laterals[-1]))

        tmps = []
        if self.cat_feats:
            t_outs = []
            pool_shape = laterals[self.shape_level].size()[2:]
            for i in range(0, self.num_outs):
                t_outs.append(self.pooling(laterals[i], pool_shape))

            t_out = torch.cat(t_outs, dim=1)
            t_out = self.fp1_convs(t_out)
            atten_feats = self.atten(t_out)
            # 6. 将one_feat pool分别得到各层的分辨率同时加上inputs
            for i in range(0, self.num_outs):
                out_size = laterals[i].size()[2:]
                # tmp=self.pooling(one_feat, output_size=out_size)
                tmp = F.interpolate(atten_feats, size=out_size, **self.upsample_cfg)
                # tmp = tmp+laterals[i]
                tmps.append(tmp)
        else:
            # attention  every layers
            tmps = [
                self.atten_layers[i](laterals[i]) for i in range(self.num_outs)
            ]

        # down to up:
        if self.down_up:
            for i in range(self.num_outs-1):
                prev_shape = tmps[i + 1].shape[2:]
                tmps[i + 1] = tmps[i + 1] + F.interpolate(
                    tmps[i],size=prev_shape, **self.upsample_cfg)

        outs = [
            self.fpl_convs[i](tmps[i]) for i in range(self.num_outs)
        ]
        return tuple(outs)