"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .deim import DEIM
from .hybrid_encoder import HybridEncoder
from .dfine_decoder import DFINETransformer

from .matcher import HungarianMatcher
from .postprocessor import PostProcessor
from .deim_criterion import DEIMCriterion