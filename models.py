import torch.nn as nn
from torchvision.models import *


def resnet18_cifar100():
    model = resnet18(weights=None)

    # Adapt for CIFAR (32x32)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    # Replace classifier since cifar100 has 100 classes
    model.fc = nn.Linear(model.fc.in_features, 100)

    return model
