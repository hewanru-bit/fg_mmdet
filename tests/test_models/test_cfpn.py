
import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.models.necks import (FPN, CFPN)

def test_neck_cfpn():
    # num_scales, in_channels, out_channels must be same length
    with pytest.raises(AssertionError):
        neck =CFPN(in_channels=[8, 16, 32, 64],
             out_channels=16,
             num_outs=4,
             shape_level=2)
        feats = (torch.rand(1, 8, 24, 24), torch.rand(1, 16, 20, 20),
                 torch.rand(1, 32, 16, 16),torch.rand(1, 64, 12, 12))
        out=neck(feats)
