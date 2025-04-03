import torch
from torch import nn
from src.core import register

__all__ = ['play', 'MLP']

@register
def play(a: int, b: int = 2):
    out = a + b
    return out

@register
class MLP(nn.Module):
    def __init__(self, in_channels = 240, hidden_dim = 500, out_channels = 10):
        super().__init__()
        self.layer_1 = nn.Linear(in_channels, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, out_channels)
    
    def forward(self, x):
        out = self.layer_2(self.layer_1(x))
        return out

'''
yaml_file_path = 'rt_detr_v1/configs/test_1.yml'
cfg = YAMLConfig(yaml_file_path)
mlp_instance = cfg.model
print(mlp_instance)
'''