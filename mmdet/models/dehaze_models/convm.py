import torch
import torch.nn as nn
import math
import warnings
from ..builder import DEHAZEMODELS
from .base_dehaze import BaseDehaze
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.utils.parrots_wrapper import _BatchNorm


@DEHAZEMODELS.register_module()
class CONVMNet(BaseDehaze):

    def __init__(self,
                 in_channels=(1, 256, 256, 512, 1024),
                 out_channels=(256, 256, 512, 1024, 2048),
                 num_convm=5,  ####构造5个CONVM
                 out_indices=(1, 2, 3, 4),   ###输出层数从0开始
                 strides=(1,1, 2, 2, 2),
                 kernel_size=(1, 3, 3, 3, 3),
                 padding=(1, 1, 1, 1, 1),
                 bias=(True, True, True, True, True),
                 act_cfg=dict(type='ReLU'),
                 plugins=None,
                 pretrained=None,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 norm_eval=False,
                 ):
        super(CONVMNet, self).__init__(init_cfg=init_cfg)

        assert plugins is None, 'Not implemented yet.'
        self.act_cfg = act_cfg
        self.num_convm = num_convm
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_activation = act_cfg is not None
        self.kernel_size = kernel_size
        self.strides = strides
        self.bias = bias
        self.padding = padding
        self.norm_eval = norm_eval

        if self.init_layer:
            self._initialize_weights()

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                act_cfg_.setdefault('inplace', True)
            self.activate = build_activation_layer(act_cfg_)

        self.features=nn.ModuleList()
        for i in range(self.num_convm):
            l_conv = ConvModule(
                in_channels=self.in_channels[i], out_channels=self.out_channels[i],
                kernel_size=self.kernel_size[i], stride=self.strides[i],
                padding=self.padding[i], bias=self.bias[i], act_cfg=self.act_cfg)
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
        for i in range(self.num_convm):
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




