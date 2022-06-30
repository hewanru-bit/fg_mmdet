# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS

def fc_unit(inchannel,outchannel):
    fc = nn.Sequential(
            nn.Linear(inchannel,128),
            nn.ReLU(),
            nn.Linear(128,outchannel)
        )
    return fc

@NECKS.register_module()
class CFPN(BaseModule):
    """CFPN ( Cross-Layer Feature Aggregation+Cross-Layer Feature Distribution)

    paper: `Cross-layer feature pyramid network for salient object detection
    <https://dx.doi.org/10.1109/TIP.2021.3072811>`_.

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        stride (int): stride of 3x3 convolutional layers
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,# [256, 512, 1024, 2048]
                 out_channels,  # 256
                 num_outs=4,
                 shape_level=2,  #平均池化的尺寸与这一层相同
                 pooling_type='AVG',
                 used_backbone_levels=4, # (0,used_backbone_levels)的层输出
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 no_norm_on_lateral=False,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CFPN, self).__init__(init_cfg)
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
        self.used_backbone_levels=used_backbone_levels

        self.la1_convs = nn.ModuleList()
        self.fp1_convs = nn.ModuleList()
        self.la2_convs = nn.ModuleList()
        self.fp2_convs = nn.ModuleList()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(self.out_channels*self.num_ins , self.num_ins, 1)
                                )
        for i in range(self.num_outs):
            l1_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            f1_conv = ConvModule(
                in_channels[i],
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            l2_conv = ConvModule(
                out_channels,
                out_channels,
                1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            f2_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.la1_convs.append(l1_conv)
            self.fp1_convs.append(f1_conv)
            self.la2_convs.append(l2_conv)
            self.fp2_convs.append(f2_conv)

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.adaptive_avg_pool2d  ##自适应全局平均池

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_ins

        outs = []
        pool_h,pool_w = inputs[self.shape_level].size()[2:]
        # 用 nn.AdaptiveAvgPool2d(1)+ 1×1卷积代替全连接层
        # self.fc = fc_unit(self.out_channels*self.num_ins*pool_h*pool_w,5)

        # 1.各个层分别进行平均池化,size为第shape_level的size
        for i in range(0, self.num_ins):
            outs.append(self.pooling(self.la1_convs[i](inputs[i]),(pool_h,pool_w)))
        # 2. 池化后的结果按照通道拼接
        out = torch.cat(outs, dim=1)
        # 3.经过二次FC得到各层的权重
        # out=out.view(-1)  # 如果使用全连接层linear需要
        ws = self.fc(out)
        ws=ws.view(-1)  #  如果使用全连接层linear不需要
        # 4.每层与对应的权重进行相乘
        inner_outs=[]
        for i in range(0, self.num_outs):
            inner_outs.append(inputs[i]*ws[i])

        # 5. pool分别得到各层

        tmp_outs = []
        for i in range(0, self.num_outs):
            out_size = inputs[i].size()[2:]
            tmp_out=self.pooling(inner_outs[i], output_size=out_size)
            tmp_out = self.fp1_convs[i](tmp_out)
            tmp_out=self.la2_convs[i](tmp_out)
            tmp_outs.append(tmp_out)

        # 6.fpn up to down
        for i in range(self.num_outs-1,0,-1):
            prev_shape = tmp_outs[i - 1].shape[2:]
            tmp_outs[i - 1] = tmp_outs[i - 1] + F.interpolate(
                tmp_outs[i],size=prev_shape, **self.upsample_cfg)
        # 7.指定使用的层,经过卷积存储
        outs = [
            self.fp2_convs[i](tmp_outs[i]) for i in range(self.used_backbone_levels)
        ]
        return tuple(outs)
