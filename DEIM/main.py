import json
from engine.core import YAMLConfig, yaml_utils
from engine.core import GLOBAL_CONFIG
import torch
 

yml_path = 'configs/test.yml'
cfg = YAMLConfig(yml_path)
print(GLOBAL_CONFIG)
train_dataloder = cfg.train_dataloader
samples, targets = next(iter(train_dataloder))
print(samples.shape)
for i, v in enumerate(targets):
    print(f'target_{i} got labels: {v["labels"]}')
    
# backbone = cfg.model
# out_backbone = backbone(samples)
     
# for i in out_backbone:
#     print(i.shape)


S3 = torch.rand(size=(2, 384, 80, 80))
S4 = torch.rand(size=(2, 768, 40, 40))
S5 = torch.rand(size=(2, 1536, 20, 20))

# # N model
# S4 = torch.rand(size=(2, 512, 40, 40))
# S5 = torch.rand(size=(2, 1024, 20, 20))

backbone_out = [S3, S4, S5]

hybrid_encoder = cfg.model
out_hybrid_encoder = hybrid_encoder(backbone_out)
    
for i in out_hybrid_encoder:
    print(i.shape)