# -*-coding:utf-8-*-
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class ACFPN(BaseModule):
    """ASFF 类似
    先进行fpn中的up to down，并如果输出不够指定层数，增加额外的层，得到通道数相同的num_outs个层。然后选择shape_level（不止一个）进行缩放
    torch.cat得到权重，每层乘以权重得到一层，注意只使用原始的输入层进行shape操作，
    增加的额外的层进行cat求权重，最后并不是使用，额外层最后还是额外层。
    down_up参数控制是否进行down to up
    """

    def __init__(self,
                 in_channels,# [256, 512, 1024, 2048]
                 out_channels,  # 256
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 down_up=False, ####
                 shape_level=[0,1,2],  #平均池化的尺寸与这一层相同
                 pooling_type='AVG',
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 no_norm_on_lateral=False,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ACFPN, self).__init__(init_cfg)
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

        inner_outs = copy.copy(laterals)
        # 3.遍历选择的shape_level层分别进行平均池化,然后进行torch.cat得到权重，权重*laterals得到一层
        for k in range (len(self.shape_level)):
            tems = []
            pool_shape= laterals[self.shape_level[k]].size()[2:]
            # 各个层分别进行平均池化,size为第shape_level的size
            for i in range(0, self.num_outs):
                tems.append(self.pooling(laterals[i],pool_shape))
            # 2. 池化后的结果按照通道拼接
            tem = torch.cat(tems, dim=1)
            # 3.经过二次FC得到各层的权重
            ws = self.fc(tem)
            ws = torch.softmax(ws, dim=1)  # # 映射到0 -1 范围
            # # 对ws 按照通道数进行分离 b,1,1,1
            w = torch.split(ws, 1, dim=1)
            # 4.每层与对应的权重进行相乘再相加得到一层
            # range(0, self.num_ins)只有对应的原始输出层的laterals是各层权重求和得到，额外添加的层不变
            inner_outs[k] = sum([tems[i]*w[i] for i in range (self.num_outs)])

        # 如果down_up is ture 就进行down to up
        if self.down_up:
            for i in range(self.num_outs-1):
                prev_shape = inner_outs[i + 1].shape[2:]
                inner_outs[i + 1] = inner_outs[i + 1] + F.interpolate(
                    inner_outs[i],size=prev_shape, **self.upsample_cfg)
        # 7.指定使用的层,经过卷积存储
        # part 1: from original levels
        outs = [
            self.fpl_convs[i](inner_outs[i]) for i in range(self.num_outs)
        ]

        return tuple(outs)
