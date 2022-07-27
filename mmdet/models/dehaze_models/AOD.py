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

@DEHAZEMODELS.register_module()
class AODNet(BaseDehaze):
    '''AODNet

    Args:
        in_channels(int): Number of input image channels. Default(1,1,2,2,4).
        base_channels(int):每个卷积层的输入通道数的基础数字，默认为3,输出通道默认为 3×(1,1,2,2,4).
        out_channels(int):每个卷积层的输出通道数，默认为3
        num_stages(int): 卷积模块个数，默认为5
        strides(Sequence[int]):卷积核步长，默认(1,1,1,1,1)
        kernel_size(Sequence[int]):卷积核尺寸，默认(1,3,5,7,3)
        bias(bool):卷积偏移，默认(True,True,True,True,True)
        act_cfg(dict | None):激活函数的配置，默认dict(type='Relu')
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        plugins (dict): plugins for convolutional layers. Default: None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    '''
    def __init__(self,
                 in_channels=(1,1,2,2,4),
                 base_channels=3,
                 out_channels=(3,3,3,3,3),
                 num_stages=5,
                 strides=(1,1,1,1,1),
                 kernel_size=(1,3,5,7,3),
                 padding=(0,1,2,3,1),
                 bias=(True,True,True,True,True),
                 act_cfg=dict(type='ReLU'),
                 plugins=None,
                 pretrained=None,
                 init_cfg=None,
                 norm_eval=False,
                 **kwargs
                 ):
        super(AODNet, self).__init__(init_cfg=init_cfg,**kwargs)

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
        assert len(in_channels) == num_stages, \
            'The length of in_channels should be equal to num_stages, ' \
            f'while the in_channels is {in_channels}, the length of ' \
            f'in_channels is {len(in_channels)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(out_channels) == num_stages, \
            'The length of out_channels should be equal to num_stages, ' \
            f'while the out_channels is {out_channels}, the length of ' \
            f'out_channels is {len(out_channels)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages, ' \
            f'while the strides is {strides}, the length of ' \
            f'strides is {len(strides)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(padding) == num_stages, \
            'The length of padding should be equal to num_stages, ' \
            f'while the padding is {padding}, the length of ' \
            f'padding is {len(padding)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(kernel_size) == num_stages, \
            'The length of kernel_size should be equal to num_stages, ' \
            f'while the kernel_size is {kernel_size}, the length of ' \
            f'kernel_size is {len(kernel_size)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(bias) == (num_stages), \
            'The length of bias should be equal to (num_stages-1), ' \
            f'while the bias is {bias}, the length of ' \
            f'bias is {len(bias)}, and the num_stages is ' \
            f'{num_stages}.'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_stages = num_stages
        self.strides = strides
        self.base_channels = base_channels
        self.bias =bias
        self.padding = padding
        self.with_activation = act_cfg is not None
        self.norm_eval = norm_eval
        self.act_cfg=act_cfg
        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                act_cfg_.setdefault('inplace', True)
            self.activate = build_activation_layer(act_cfg_)


        self.CONVM = nn.ModuleList()
        for i in range(self.num_stages):
            conv_act=ConvModule(
                in_channels=self.in_channels[i] * base_channels, out_channels=self.out_channels[i],
                kernel_size=self.kernel_size[i], stride=self.strides[i],
                padding=self.padding[i], bias=self.bias[i],act_cfg=self.act_cfg)
            self.CONVM.append(conv_act)

        self.conv_1_1 = nn.Conv2d(256,1,3,1,1)

    '''为了B系列，在fpn之后，head之前求loss,并不经过dehaze_model,把forward重写
     只改变尺寸和求loss,新加一个卷积层conv_1_1来改变通道数'''

    def forward(self, x):
        # # 前两层正常经过，之后每一层与前一层 concat之后经过卷积层，最后一层将前面所有层的输出concat之后经过最后一层
        # outs=[]
        # x1=x
        # for i in range(self.num_stages):
        #     if i>1 and i!=(self.num_stages-1):###i=2时，开始 concat
        #         x1 = torch.cat((outs[i - 2], outs[i - 1]),1)
        #
        #     if i==self.num_stages-1:  #最后一层所有都拼接起来
        #         x1= torch.cat([outs[j] for j in range(len(outs))],1)
        #
        #     x1 = self.CONVM[i](x1)  # x0,x1 正常经过卷积层
        #     outs.append(x1)
        # result = self.activate((outs[-1]*x)-outs[-1]+1)

        '''为了B系列，在fpn之后，head之前求loss,并不经过dehaze_model,把forward重写
             只改变尺寸和求loss,新加一个卷积层conv_1_1来改变通道数'''
        result = self.conv_1_1(x) # 改变通道数

        return result



    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(AODNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def loss(self, dehaze_img, gt_img, img_metas):
        if isinstance(self.loss_dehaze, list):
            losses = []
            for loss_dehaze in self.loss_dehaze:
                loss = loss_dehaze(dehaze_img, gt_img)
                if isinstance(loss, tuple):
                    # Perceptual Loss
                    loss = loss[0]
                losses.append(loss)
        else:
            losses = self.loss_dehaze(dehaze_img, gt_img)
        if self.loss_perseptual is not None:
            device = dehaze_img.device
            loss_perseptual = self.loss_perseptual.to(device)
            perseptual_loss, style_loss = loss_perseptual(dehaze_img, gt_img)
            return dict(branch_loss=losses, perseptual_loss=perseptual_loss)
        else:
            return dict(branch_loss=losses)

    def get_results(self, *args, **kwargs):
        pass


    '''为了B系列，在fpn之后，head之前求loss,并不经过dehaze_model,把forward重写
     只改变尺寸和求loss'''
    def forward_train(self,
                      img,
                      img_metas,
                      gt_img,
                      return_dehaze_img=False,
                      **kwargs):
        """
        Args:
            img (Tensor): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_img (Tensor): Ground truth images, has a shape
                (num_gts, H, W).

        Returns:
            losses: (dict): A dictionary of loss components.
        """
        edge_img = self(img) # 改变了通道数
        # 对gt_img下采样到edge_img 的尺寸
        w, h = edge_img.size()[2:]
        gt_img = F.interpolate(gt_img, size=(w, h))

        assert gt_img is not None
        assert edge_img.shape == gt_img.shape
        losses = self.loss(
            *(edge_img, ), gt_img=gt_img, img_metas=img_metas)

        if return_dehaze_img:
            return edge_img, losses
        else:
            return losses






