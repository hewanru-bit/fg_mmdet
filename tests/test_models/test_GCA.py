import mmcv
import torch

from mmdet.models.dehaze_models import GCANet

def test_anchor_head_loss():
    """Tests anchor head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    cfg = mmcv.Config(
        dict(
            plugins=None,
            pretrained=None,
            init_cfg=None,
            only_residual=True,))
    self = GCANet(**cfg)
    x = torch.rand(1,3,256,256)
    cls_scores = self.forward(x)
    print(cls_scores.shape)