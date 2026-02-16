import torch.nn as nn
from torchvision.models import resnet18

class ResNet18CIFAR(nn.Module):
    def __init__(self,num_classes=100,dropout=0.0):
        super().__init__()
        m=resnet18(weights=None)
        m.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        m.maxpool=nn.Identity()
        in_f=m.fc.in_features
        m.fc=nn.Identity()
        self.backbone=m
        self.head=nn.Sequential(nn.Dropout(p=float(dropout)) if dropout and dropout>0 else nn.Identity(),nn.Linear(in_f,num_classes))
    def forward(self,x):
        x=self.backbone(x)
        return self.head(x)

def resnet18_cifar100(dropout=0.0):
    return ResNet18CIFAR(num_classes=100,dropout=dropout)
