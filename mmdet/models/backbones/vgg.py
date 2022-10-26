import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    11: 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    13: 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    16: 'https://download.pytorch.org/models/vgg16-397923af.pth',
    19: 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):
    arch_settings = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
    def __init__(self,
                 depth=16,
                 in_channels=3,
                 base_channels=64,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 pretrained=True,
                 batch_norm=None,
                 ):
        super(VGG, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        self._initialize_weights()
        self.depth =depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.batch_norm = batch_norm
        self.pretrained = pretrained
        self.layers=[]
        self._make_layers()
        

    def forward(self, x):
        x = self.layers(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layers(self):
        layer = []
        in_channels = 3
        for v in self.arch_settings[self.depth]:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if self.batch_norm:
                layer += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layer += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            if v == 'M':
                layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
                self.layers.append(nn.Sequential(*layer))
                layer =[]

        return self.layers


def vgg11(pretrained=False, model_root=None, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = VGG(make_layers(arch_settings['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[self.depth]))
    return model

if __name__ == '__main__':
    input = torch.randn(2, 3, 64, 64)
    se = VGG()
    output = se(input)
    print(output.shape)