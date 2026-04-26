import torch
import torch.nn.init as init
import torch.nn as nn
from torchvision import models

def get_backbone(num_classes=10):
    model = models.resnet18(weights=None)
    
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model