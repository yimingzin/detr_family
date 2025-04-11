"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .common import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
from .presnet import PResNet

from .timm_model import TimmModel
from .torchvision_model import TorchVisionModel


from .hgnetv2 import HGNetv2
