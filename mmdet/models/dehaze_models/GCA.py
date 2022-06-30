import torch
import torch.nn as nn
import math
import warnings
from ..builder import DEHAZEMODELS
from .base_dehaze import BaseDehaze
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.utils.parrots_wrapper import _BatchNorm
import torch.nn.functional as F

class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)


class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(SmoothDilatedResidualBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
        y = self.norm2(self.conv2(self.pre_conv2(y)))
        return F.relu(x+y)


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)
        nn.MaxPool2d(3, 2, 1)
    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)


@DEHAZEMODELS.register_module()
class GCANet(BaseDehaze):
    def __init__(self,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None,
                 norm_eval=False,
                 only_residual=True,
                 ):
        super(GCANet, self).__init__(init_cfg=init_cfg)

        self.norm_eval = norm_eval

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        assert plugins is None, 'Not implemented yet.'

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)

        self.res1 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res2 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res3 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res4 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res5 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res6 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res7 = ResidualBlock(64, dilation=1)

        self.gate = nn.Conv2d(64 * 3, 3, 3, 1, 1, bias=True)

        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.norm4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(64, affine=True)
        self.deconv1 = nn.Conv2d(64, 3, 1)
        self.only_residual = only_residual

        # self.conv_J_1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        # self.conv_J_2 = nn.Conv2d(64, 3, 3, 1, 1, bias=False)
        self.conv_T_1 = nn.Conv2d(64, 16, 3, 1, 1, bias=False)
        self.conv_T_2 = nn.Conv2d(16, 3, 3, 1, 1, bias=False)


    def forward(self, x, Val=True):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y1 = F.relu(self.norm3(self.conv3(y)))

        y = self.res1(y1)
        y = self.res2(y)
        y = self.res3(y)
        y2 = self.res4(y)
        y = self.res5(y2)
        y = self.res6(y)
        y3 = self.res7(y)

        gates = self.gate(torch.cat((y1, y2, y3), dim=1))
        out = y1 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :]
        out = F.relu(self.norm4(self.deconv3(out)))
        out_J = F.relu(self.norm5(self.deconv2(out)))
        if self.only_residual:
            out_J = self.deconv1(out_J)
        else:
            out_J = F.relu(self.deconv1(out_J))
        out_J = out_J + (x[:, :3] + 128.0)
        # out_J = self.conv_J_1(out)
        # out_J = self.conv_J_2(out_J)
        # out_J = F.upsample(out_J, x.size()[2:], mode='bilinear')
        # out_J = out_J + x[:, :3]

        out_T = self.conv_T_1(out)
        out_T = self.conv_T_2(out_T)
        out_T = F.upsample(out_T, out_J.size()[2:], mode='bilinear')
        if Val == False:
            return out_J
        else:
            return out_T



    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(GCANet, self).train(mode)
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









