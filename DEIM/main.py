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

###############################################    backbone Test    ###############################################

# backbone = cfg.model
# out_backbone = backbone(samples)
     
# for i in out_backbone:
#     print(i.shape)


###############################################    hybridEncoder Test    ###############################################

# S3 = torch.rand(size=(2, 384, 80, 80))
# S4 = torch.rand(size=(2, 768, 40, 40))
# S5 = torch.rand(size=(2, 1536, 20, 20))

# # # N model
# # S4 = torch.rand(size=(2, 512, 40, 40))
# # S5 = torch.rand(size=(2, 1024, 20, 20))

# backbone_out = [S3, S4, S5]

# hybrid_encoder = cfg.model
# out_hybrid_encoder = hybrid_encoder(backbone_out)
    
# for i in out_hybrid_encoder:
#     print(i.shape)

###############################################    decoder Test    ###############################################


# mock_feat1 = torch.rand(size=(4, 256, 80, 80)) # H/8, W/8
# mock_feat2 = torch.rand(size=(4, 256, 40, 40)) # H/16, W/16
# mock_feat3 = torch.rand(size=(4, 256, 20, 20)) # H/32, W/32
# decoder_input_feats = [mock_feat1, mock_feat2, mock_feat3]

# decoder_instance = cfg.model
# decoder_outputs = decoder_instance(decoder_input_feats, targets=targets)


###############################################    DEIM Test    ###############################################

model_instance = cfg.model
outputs = model_instance(samples, targets=targets)

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