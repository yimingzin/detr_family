import json
from src.core import YAMLConfig
from src.core import GLOBAL_CONFIG
import data.dataloader
from play_model import *
import torch

if __name__ == '__main__':
    yml_path = 'rt_detr_v1/configs/test_1.yml'
    cfg = YAMLConfig(yml_path)
    print(GLOBAL_CONFIG)
    train_dataloder = cfg.train_dataloader
    samples, targets = next(iter(train_dataloder))
    print(samples.shape)
    print(targets[0])
