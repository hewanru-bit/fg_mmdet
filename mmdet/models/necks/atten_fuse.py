# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from ..builder import NECKS
from mmdet.models.attention import SEAttention
from mmcv.cnn import build_norm_layer

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=False):
        super(SeparableConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = relu

        self.sep = nn.Conv2d(in_channels,
                             in_channels,
                             3,
                             padding=1,
                             groups=in_channels,
                             bias=False)
        self.pw = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            bias=bias)
        if relu:
            self.relu_fn = Swish()

    def forward(self, x):
        x = self.pw(self.sep(x))
        if self.relu:
            x = self.relu_fn(x)
        return x

class WeightedInputConv_V2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_ins,
                 conv_cfg=None,
                 norm_cfg=None,
                 separable_conv=True,
                 act_cfg=None,
                 eps=0.0001):
        super(WeightedInputConv_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_ins = num_ins
        self.eps = eps
        if separable_conv:
            _, bn_layer = build_norm_layer(norm_cfg, out_channels)
            self.conv_op = nn.Sequential(
                SeparableConv(
                    in_channels,
                    out_channels,
                    bias=True,
                    relu=False),
                bn_layer
            )
        else:
            self.conv_op = ConvModule(
                in_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=None,
                inplace=False)

        # edge weight and swish
        self.weight = nn.Parameter(torch.Tensor(self.num_ins).fill_(1.0))
        self._swish = Swish()

    def forward(self, inputs):
        assert isinstance(inputs, list)
        assert len(inputs) == self.num_ins
        w = F.relu(self.weight)
        w /= (w.sum() + self.eps)
        x = 0
        for i in range(self.num_ins):
            x += w[i] * inputs[i]
        # import pdb; pdb.set_trace()
        output = self.conv_op(self._swish(x))
        return output


@NECKS.register_module()
class ATTENFUSE(BaseModule):
    """
    注意力机制，特征融合，fpn之后
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 down_up=False,  ####
                 cat_feats=False,
                 num_outs =5,
                 shape_level=2,  # 平均池化的尺寸与这一层相同
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ATTENFUSE, self).__init__(init_cfg)
        # assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.down_up = down_up
        self.cat_feats =cat_feats
        self.shape_level = shape_level
        self.num_outs = num_outs

        self.fpl_convs = nn.ModuleList()
        self.atten_layers = nn.ModuleList()

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
                channel = out_channels, reduction= 16
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
            #############################
            self.weigtconv = WeightedInputConv_V2( in_channels=out_channels,
                 out_channels=out_channels,
                 num_ins=self.num_outs,)

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.adaptive_avg_pool2d  ##自适应全局平均池


    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""

        # attention
        atten_feats = [
            self.atten_layers[i](inputs[i]) for i in range(self.num_outs)
        ]

        if self.cat_feats:
            t_outs = []
            pool_shape = inputs[self.shape_level].size()[2:]
            for i in range(0, self.num_outs):
                t_outs.append(self.pooling(inputs[i], pool_shape))

            one_feat = torch.cat(t_outs, dim=1)
            # 将one_feat channels b*num_out--->b
            one_feat = self.fp1_convs(one_feat)

            # # 使用可学习的参数weightinputsconv得到one_feats
            # one_feat = self.weigtconv(t_outs)

            # one_feat pool分别得到各层的分辨率同时加上inputs
            for i in range(0, self.num_outs):
                out_size = inputs[i].size()[2:]
                # 双线性插值回去
                atten_feats[i] = F.interpolate(one_feat,size=out_size, **self.upsample_cfg)

        # down to up:
        if self.down_up:
            for i in range(self.num_outs-1):
                prev_shape = atten_feats[i + 1].shape[2:]
                atten_feats[i + 1] = atten_feats[i + 1] + F.interpolate(
                    atten_feats[i],size=prev_shape, **self.upsample_cfg)

        outs = [
            self.fpl_convs[i](atten_feats[i]) for i in range(self.num_outs)
        ]
        return tuple(outs)