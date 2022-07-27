# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class BCFPN(BaseModule):
    """
    先进行fpn中的up to down，并如果输出不够指定层数，增加额外的层，得到通道数相同的num_outs个层。
    即outs = inputs+w*inputs, down_up参数控制是否进行down to up
    """

    def __init__(self,
                 in_channels,# [256, 512, 1024, 2048]
                 out_channels,  # 256
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 down_up=False, ####
                 cat_feats =False,
                 shape_level=2,  #平均池化的尺寸与这一层相同
                 pooling_type='AVG',
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 no_norm_on_lateral=False,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(BCFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.shape_level = shape_level
        self.no_norm_on_lateral = no_norm_on_lateral
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.upsample_cfg = upsample_cfg.copy()
        self.down_up = down_up
        self.relu_before_extra_convs =relu_before_extra_convs
        self.cat_feats = cat_feats

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

        self.la1_convs = nn.ModuleList()

        self.fpn_convs = nn.ModuleList()
        self.fpl_convs = nn.ModuleList()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(self.out_channels*self.num_outs, self.num_outs, 1)
                                )
        for i in range(self.start_level, self.backbone_end_level):
            l1_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.la1_convs.append(l1_conv)
        for j in range (self.num_outs):
            f2_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fl_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(f2_conv)
            self.fpl_convs.append(fl_conv)

        if self.cat_feats:
            self.fp1_convs = ConvModule(
                    out_channels*self.num_outs,
                    out_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)

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

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_ins

        # 1.首先通过1*1conv统一通道数
        laterals = [
            la1_conv(inputs[i + self.start_level])
            for i, la1_conv in enumerate(self.la1_convs)
        ]

        # 2. fpn up to down:
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]  # 得到每层laterals的H，W
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

        laterals = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # 如果输出层数要求大于输入层，增加额外层数
        if self.num_outs > len(laterals):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - self.num_ins):
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

        # 3.层分别进行平均池化,然后进行torch.cat
        t_outs = []
        pool_shape = laterals[self.shape_level].size()[2:]

        for i in range(0, self.num_outs):
            t_outs.append(self.pooling(laterals[i],pool_shape))
        t_out = torch.cat(t_outs, dim=1)
        # 4.用nn.AdaptiveAvgPool2d(1)+ 1×1卷积代替全连接层得到各层的权重
        ws = self.fc(t_out)
        ws = torch.sigmoid(ws)  # # 映射到0 -1 范围
        # # 对ws 按照通道数进行分离 b,1,1,1
        w = torch.split(ws, 1, dim=1)
        inner_outs = []
        # 如果得到one_feature再进行pooling
        if self.cat_feats:
            for i in range(0, self.num_outs):
                inner_outs.append(t_outs[i]*w[i])
            # torch .cat 得到一层特征，用于之后的poling
            one_feat = torch.cat(inner_outs,dim=1)
            # 将one_feat channels b*num_out--->b
            one_feat = self.fp1_convs(one_feat)
            # 6. 将one_feat pool分别得到各层的分辨率同时加上inputs
            tmps = []
            for i in range(0, self.num_outs):
                out_size = laterals[i].size()[2:]
                tmp=self.pooling(one_feat, output_size=out_size)
                # tmp = tmp+laterals[i]
                tmps.append(tmp)
        else:
            # 每层与对应的权重进行相乘
            for i in range(0, self.num_outs):
                # inner_outs.append(laterals[i]*w[i]+laterals[i])
                inner_outs.append(laterals[i] * w[i])
            tmps = inner_outs

        # 如果down_up is ture 就进行down to up
        if self.down_up:
            for i in range(self.num_outs-1):
                prev_shape = tmps[i + 1].shape[2:]
                tmps[i + 1] = tmps[i + 1] + F.interpolate(
                    tmps[i],size=prev_shape, **self.upsample_cfg)

                # 增加原来的特征 identity
                # tmps[i + 1] = laterals[i + 1] + tmps[i + 1] + F.interpolate(
                #     tmps[i], size=prev_shape, **self.upsample_cfg)

        # 7.指定使用的层,经过卷积存储
        # part 1: from original levels
        outs = [
            self.fpl_convs[i](tmps[i]) for i in range(self.num_outs)
        ]

        return tuple(outs)
