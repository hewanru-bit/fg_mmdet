
import torch.nn.functional as F
import torch
import torch.nn as nn
import warnings
from ..builder import DEHAZEMODELS
from .base_dehaze import BaseDehaze
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.utils.parrots_wrapper import _BatchNorm

class BlockUNet1(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, relu=False, drop=False, bn=True):
        super(BlockUNet1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

        self.dropout = nn.Dropout2d(0.5)
        self.batch = nn.InstanceNorm2d(out_channels)

        self.upsample = upsample
        self.relu = relu
        self.drop = drop
        self.bn = bn

    def forward(self, x):
        if self.relu == True:
            y = F.relu(x)
        elif self.relu == False:
            y = F.leaky_relu(x, 0.2)
        if self.upsample == True:
            y = self.deconv(y)
            if self.bn == True:
                y = self.batch(y)
            if self.drop == True:
                y = self.dropout(y)

        elif self.upsample == False:
            y = self.conv(y)
            if self.bn == True:
                y = self.batch(y)
            if self.drop == True:
                y = self.dropout(y)

        return y


@DEHAZEMODELS.register_module()
class ANet(BaseDehaze):
    def __init__(self, in_channels=3, out_channels=3,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None,
                 norm_eval=False
                 ):
        super(ANet, self).__init__(init_cfg=init_cfg)
        self.norm_eval=norm_eval
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


        self.conv = nn.Conv2d(in_channels, 8, 4, 2, 1, bias=False)
        self.layer1 = BlockUNet1(8, 16)
        self.layer2 = BlockUNet1(16, 32)
        self.layer3 = BlockUNet1(32, 64)
        self.layer4 = BlockUNet1(64, 64)
        self.layer5 = BlockUNet1(64, 64)
        self.layer6 = BlockUNet1(64, 64)
        self.layer7 = BlockUNet1(64, 64)
        self.dlayer7 = BlockUNet1(64, 64, True, True, True, False)
        self.dlayer6 = BlockUNet1(128, 64, True, True, True)
        self.dlayer5 = BlockUNet1(128, 64, True, True, True)
        self.dlayer4 = BlockUNet1(128, 64, True, True)
        self.dlayer3 = BlockUNet1(128, 32, True, True)
        self.dlayer2 = BlockUNet1(64, 16, True, True)
        self.dlayer1 = BlockUNet1(32, 8, True, True)
        self.relu = nn.ReLU()
        self.dconv = nn.ConvTranspose2d(16, out_channels, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.layer1(y1)
        y3 = self.layer2(y2)
        y4 = self.layer3(y3)
        y5 = self.layer4(y4)
        y6 = self.layer5(y5)
        y7 = self.layer6(y6)
        y8 = self.layer7(y7)

        dy8 = self.dlayer7(y8)
        concat7 = torch.cat([dy8, y7], 1)
        dy7 = self.dlayer6(concat7)
        concat6 = torch.cat([dy7, y6], 1)
        dy6 = self.dlayer5(concat6)
        concat5 = torch.cat([dy6, y5], 1)
        dy5 = self.dlayer4(concat5)
        concat4 = torch.cat([dy5, y4], 1)
        dy4 = self.dlayer3(concat4)
        concat3 = torch.cat([dy4, y3], 1)
        dy3 = self.dlayer2(concat3)
        concat2 = torch.cat([dy3, y2], 1)
        dy2 = self.dlayer1(concat2)
        concat1 = torch.cat([dy2, y1], 1)
        out = self.relu(concat1)
        out = self.dconv(out)
        out = self.lrelu(out)

        result = F.avg_pool2d(out, (out.shape[2], out.shape[3]))

        return result


def train(self, mode=True):
    """Convert the model into training mode while keep normalization layer
    freezed."""
    super(ANet, self).train(mode)
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




