import torch
import torchvision.models
from torchvision import models
from torch import nn
from torchvision.transforms import Resize
from timm.data.mixup import Mixup

x = torch.randn(1, 3, 896, 896)
f = Resize(size=(224, 224))
labels = torch.randn(1, 7)
x = f(x)
model = torchvision.models.swin_v2_b()
mixup_fn = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
    prob=0.1, switch_prob=0.5, mode='batch',
    label_smoothing=0.1, num_classes=7)
samples, targets = mixup_fn(x, labels)
print(samples.shape())
print(targets.shape())
# model = models.maxvit_t(weights='MaxVit_T_Weights.IMAGENET1K_V1')
# model.classifier = nn.Sequential(
#     nn.AdaptiveAvgPool2d(1),
#     nn.Flatten(),
#     nn.LayerNorm(512),
#     nn.Linear(512, 512),
#     nn.Tanh(),
#     nn.Linear(512, 7, bias=False),
# )
# features = model(samples)
# print(features)
