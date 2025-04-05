import json
from src.core import YAMLConfig
from src.core import GLOBAL_CONFIG
import data.dataloader
from play_model import *
from src.nn import *
from src.zoo import *
import torch

if __name__ == '__main__':
    yml_path = 'rt_detr_v1/configs/test_1.yml'
    cfg = YAMLConfig(yml_path)
    train_dataloder = cfg.train_dataloader
    samples, targets = next(iter(train_dataloder))
    print(samples.shape)
    # print(targets)
    for i in targets:
        print(i['labels'])
    
    '''
    backbone = cfg.model
    print(GLOBAL_CONFIG)
    out_backbone = backbone(samples)
    '''
    
    S3 = torch.rand(size=(4, 512, 80, 80))
    S4 = torch.rand(size=(4, 1024, 40, 40))
    S5 = torch.rand(size=(4, 2048, 20, 20))

    sim_backbone_out = [S3, S4, S5]
    
    hybrid_encoder = cfg.model
    print(GLOBAL_CONFIG)
    out_hybrid_encoder = hybrid_encoder(sim_backbone_out)
    
    for i in out_hybrid_encoder:
        print(i.shape)