import json
from src.core import YAMLConfig
from src.core import GLOBAL_CONFIG
import data.dataloader
from play_model import *
from src.nn import *
from src.zoo import *
import torch


yml_path = 'rt_detr_v1/configs/test_1.yml'
cfg = YAMLConfig(yml_path)
train_dataloder = cfg.train_dataloader
samples, targets = next(iter(train_dataloder))
print(samples.shape)
# print(targets)
for i in targets:
    print(i['labels'])
    
    
'''
###############################################    backbone Test    ###############################################
backbone = cfg.model
print(GLOBAL_CONFIG)
out_backbone = backbone(samples)
'''
    
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
'''
    

model_instance = cfg.model
outputs = model_instance(samples, targets = targets)
    
    
###############################################    decoder Test    ###############################################
for k, v in outputs.items():
    print(f"\nKey: '{k}'") 

    if isinstance(v, torch.Tensor):
        print(f"  Type: Tensor")
        print(f"  Shape: {v.shape}")

    elif isinstance(v, list):
        print(f"  Type: List")
        print(f"  Length: {len(v)}")
        if len(v) > 0 and isinstance(v[0], dict):
            print(f"  List contains dicts. Example element 0 keys: {list(v[0].keys())}")
            print("    Shapes inside first dict element:")
            for inner_key, inner_value in v[0].items():
                if isinstance(inner_value, torch.Tensor):
                    print(f"      '{inner_key}': {inner_value.shape}")
                else:
                    print(f"      '{inner_key}': Type {type(inner_value)}")
        elif len(v) > 0:
                print(f"  List contains elements of type: {type(v[0])}")


    elif isinstance(v, dict):
        print(f"  Type: Dictionary")
        print(f"  \nKeys: {list(v.keys())}")
        # print(f"  Content: {v}")

    else:
        # 其他类型
        print(f"  Type: {type(v)}")
        # print(f"  Value: {v}")
    
    
'''
###############################################    matcher Test    ###############################################
criterion_instance = cfg.criterion
matcher_idx = criterion_instance(outputs, targets)
for i, idx in enumerate(matcher_idx):
    pred_indices = idx[0]
    target_indices = idx[1]
    original_gt_labels = targets[i]['labels']
        
    print(f'target_{i}: {idx} | origin target: {original_gt_labels[target_indices]}')
'''
    
    
criterion_instance = cfg.criterion
loss = criterion_instance(outputs, targets)
print(loss)