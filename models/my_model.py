"""
Example model. 

Author: Jinhui Yi
Date: 2023.06.01
"""
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from utils.pytorch_misc import *


class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.num_classes = cfg.num_classes
        # self.model_name = 'resnet50'
        # self.model_name = 'swin_v2_s'
        self.model_name = 'swin_v2_b'  # wr: lr=1.5e-4,
        # self.model_name = 'swin_v2_l'
        # self.model_name = 'convnext_s'
        # self.model_name = 'convnext_l'
        # self.model_name = 'maxvit_t'
        # self.model_name = 'maxvit_b'
        # self.model_name = 'coatnet_b'
        # self.model_name = 'coatnet_t'
        # self.model_name = 'vit_l'
        # self.model_name = 'maxvit_l'
        # self.model_name = 'model_ensemble'
        # assert self.model_name in models.list_models()

        print("Loading pretrained: ", self.model_name)
        self.model = getattr(models, self.model_name)(weights='DEFAULT')
        # self.model = models.convnext_small(pretrain=True, num_class=cfg.num_classes)
        # self.model = models.convnext_large(pretrain=True, num_class=cfg.num_classes)
        # self.model = models.maxvit_t(weights='MaxVit_T_Weights.IMAGENET1K_V1')
        # self.model.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.LayerNorm(512),
        #     nn.Linear(512, 512),
        #     nn.Tanh(),
        #     nn.Linear(512, 7, bias=False)
        # ) #replace the final FC layer(maxvit_t)
        # self.model = timm.create_model(model_name='maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k', pretrained=True,
        #                                num_classes=self.num_classes)
        # self.model = timm.create_model(model_name='coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=True,
        #                                num_classes=self.num_classes)
        # self.model = timm.create_model(model_name='coatnet_2_rw_224.sw_in12k_ft_in1k', pretrained=True,
        #                               numrclasses=self.num_classes)
        # self.model = timm.create_model(model_name='vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=True,
        #                                num_classes=self.num_classes)
        # self.model = timm.create_model(model_name='maxvit_large_tf_512.in21k_ft_in1k', pretrained=True,
        #                                num_classes=self.num_classes)
        # self.model = timm.create_model(model_name='maxvit_tiny_tf_512.in1k', pretrained=True,
        #                                num_classes=self.num_classes)
        # self.model = timm.create_model(model_name='swinv2_large_window12to24_192to384.ms_in22k_ft_in1k', pretrained=True,
        #                                num_classes=self.num_classes)
        # self.model = timm.create_model(model_name='swinv2_large_window12to16_192to256.ms_in22k_ft_in1k', pretrained=True,
        #                                num_classes=self.num_classes)
        # self.model.classifier = nn.Sequential(nn.Linear(self.model.num_classes, cfg.num_classes)) #replace the final FC layer(convnet)
        # self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes) # replace the final FC layer (ResNet50)
        self.model.head = nn.Linear(self.model.head.in_features,
                                    self.num_classes)  # replace the final FC layer (swin_v2_s)

    def forward(self, x):
        return self.model(x)
