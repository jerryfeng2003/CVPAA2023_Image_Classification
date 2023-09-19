"""
Example model. 

Author: Jinhui Yi
Date: 2023.06.01
"""
import torch
import sys
import timm
import torch.nn as nn
import torchvision.models as models

from models.modeling_pretrain import pretrain_mae_base_patch16_224
from models.modeling_finetune import vit_base_patch16_224
from models.models_mae import mae_vit_base_patch16


# class MyModel(nn.Module):
#     def __init__(self, cfg):
#         super(MyModel, self).__init__()
#         self.num_classes = cfg.num_classes
#         # self.model_name = 'resnet50'
#         # self.model_name = 'swin_v2_s'
#         self.model_name = 'pretrain_mae_base_patch16_224'  # 新增的MAE网络
#         # assert self.model_name in models.list_models()
#
#         print("Loading pretrained: ", self.model_name)
#         # self.model = getattr(models, self.model_name)(weights='DEFAULT')
#         self.model = pretrain_mae_base_patch16_224(pretrained=True)
#         # self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes) # replace the final FC layer (ResNet50)
#         # self.model.head = nn.Linear(self.model.head.in_features, self.num_classes) # replace the final FC layer (swin_v2_s)
#
#     def forward(self, x):
#         return self.model(x)

class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.num_classes = cfg.num_classes

        # 定义MAE网络
        # self.mae_network = pretrain_mae_base_patch16_224(pretrained=True)
        # self.mae_network = vit_base_patch16_224(pretrained=True)
        # self.mae_network = mae_vit_base_patch16()
        self.mae_network = models.vit_l_16()
        self.mae_network.load_state_dict = torch.load(cfg.MODEL_STATE_DICT)
        # for key, value in self.mae_network.state_dict().items():
        #     new_key = key.replace("model.", "")
        #     if new_key in self.state_dict():
        #         self.state_dict()[new_key].copy_(value)
        # state_dict_mae = torch.load(cfg.MODEL_STATE_DICT)
        # state_dict_mae = {key.replace("model.", ""): value for key, value in state_dict_mae.items()}
        # self.mae_network.load_state_dict(state_dict_mae,strict=False)
        self.classifier = nn.Linear(self.mae_network.embed_dim, self.num_classes)
        # self.fc = nn.Linear(768, self.num_classes)
        # 其他模型组件定义
        # self.fc = nn.Sequential(
        #     nn.Flatten(1),
        #     nn.Linear(384, cfg.num_classes),
        # )  # 假设输出类别数为num_classes

    def forward(self, x):
        # MAE网络前向传播
        # print(x.shape, "$$$$$$$$$$$$$$")
        mae_output = self.mae_network(x)
        # print(mae_output.shape)

        # 继续其他模型组件的前向传播
        # fc_output = self.fc(mae_output[0].reshape(x.size(0), -1))
        output = self.classifier(mae_output)

        return mae_output
