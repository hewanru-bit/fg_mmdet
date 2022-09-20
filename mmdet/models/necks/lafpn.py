# -*-coding:utf-8-*-
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from ..builder import NECKS

class LayerAttention(nn.Module):

    def __init__(self,
                 inchannel=256,
                 outchannel=256,
                 unit_shape=2,
                 num_level=5,
                 upsample_cfg=dict(mode='nearest'),):
        super().__init__()
        self.unit_shape = unit_shape
        self.num_level = num_level
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.upsample_cfg= upsample_cfg

        self.atten_level = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannel * num_level, num_level, 1),
        )

    def forward(self, x):
        assert len(x) == self.num_level
        cat_feats = []
        resize_shape = x[self.unit_shape].size()[2:]
        for i in range(0, self.num_level):
            cat_feats.append(F.interpolate(x[i], size=resize_shape, **self.upsample_cfg))

        cat_feat = torch.cat(cat_feats, dim=1)
        ws = self.atten_level(cat_feat)
        ws = torch.sigmoid(ws)
        w = torch.split(ws, 1, dim=1)
        out =[]
        for i in range(0, self.num_level):
            out.append(x[i] * w[i])
        return out


@NECKS.register_module()
class LAFPN(BaseModule):
    """
    LAFPN
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 shape_level=2,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 no_norm_on_lateral=False,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(LAFPN, self).__init__(init_cfg)
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
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        self.layeratten = LayerAttention(
            inchannel=out_channels,
            outchannel=out_channels,
            unit_shape= self.shape_level,
            num_level=self.num_outs)

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

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        for i in range(self.num_outs):
            out_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.out_convs.append(out_conv)

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
        # 1. 1x1conv -->c=256
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # 2. fpn up to down:
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i-1]+ F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

        laterals = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # 3. add extra levels
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

        # 4. channelattention
        tmps = self.layeratten(laterals)

        # 5. down to up
        for i in range(self.num_outs-1):
            prev_shape = tmps[i + 1].shape[2:]
            tmps[i + 1] = tmps[i + 1] + F.interpolate(
                tmps[i],size=prev_shape, **self.upsample_cfg)
        # 6. outs

        outs = [
            self.out_convs[i](tmps[i]+laterals[i]) for i in range(self.num_outs)
        ]
        return tuple(outs)

