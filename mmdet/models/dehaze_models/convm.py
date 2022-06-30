import torch
import torch.nn as nn
import math
import warnings
import torch.utils.checkpoint as cp
from ..builder import DEHAZEMODELS
from .base_dehaze import BaseDehaze
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import BaseModule

class convdown(BaseModule):

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 dilation=1,
                 downsample=True,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(convdown, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, out_channel, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, out_channel, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channel,
            out_channel,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, out_channel, out_channel, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        if self.downsample:
         self.down = build_conv_layer(conv_cfg, out_channel, out_channel,
                                      2,stride=2,padding=1, bias=False)

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample:
            out = self.down(out)
        return out



@DEHAZEMODELS.register_module()
class CONVMNet(BaseDehaze):

    def __init__(self,
                 in_channels=(1, 256, 256, 512, 1024),
                 out_channels=(256, 256, 512, 1024, 2048),
                 num_block=5,  ####构造5个CONVM
                 out_indices=(1, 2, 3, 4),   ###输出层数从0开始
                 plugins=None,
                 pretrained=None,
                 downsample=True,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 norm_eval=False,
                 ):
        super(CONVMNet, self).__init__(init_cfg=init_cfg)

        assert plugins is None, 'Not implemented yet.'
        self.num_block = num_block
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_eval = norm_eval
        self.downsample=downsample


        self.features=nn.ModuleList()
        for i in range(self.num_block):
            l_conv = convdown(
                in_channel=self.in_channels[i], out_channel=self.out_channels[i],
                downsample=self.downsample,init_cfg=self.init_cfg)
            self.features.append(l_conv)


    # def _initialize_weights(self) -> None:
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)
    #

    def forward(self, x):
        outs=[]
        for i in range(self.num_block):
            x=self.features[i](x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(CONVMNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


    def loss(self, enhance_img, gt_img, img_metas):
        if isinstance(self.loss_enhance, list):
            losses = []
            for loss_enhance in self.loss_enhance:
                loss = loss_enhance(enhance_img, gt_img)
                if isinstance(loss, tuple):
                    # Perceptual Loss
                    loss = loss[0]
                losses.append(loss)
        else:
            losses = self.loss_enhance(enhance_img, gt_img)
        if self.loss_perseptual is not None:
            device = enhance_img.device
            loss_perseptual = self.loss_perseptual.to(device)
            perseptual_loss, style_loss = loss_perseptual(enhance_img, gt_img)
            return dict(enhance_loss=losses, perseptual_loss=perseptual_loss)
        else:
            return dict(enhance_loss=losses)

    def get_results(self, *args, **kwargs):
        pass




