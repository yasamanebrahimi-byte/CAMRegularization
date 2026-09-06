"""Small, dependency-light 3D ResNet18-style model used by the DaT workflow.

The official runtime includes MONAI, but keeping the exact model definition in
this module makes training and the packaged submission use the same weights
even when the research environment does not have MONAI installed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    """A 3D ResNet18-style classifier with adaptive global pooling."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.0,
        base_channels: int = 32,
    ):
        super().__init__()
        base = int(base_channels)
        if base <= 0 or in_channels <= 0 or num_classes <= 0:
            raise ValueError("in_channels, num_classes, and base_channels must be positive.")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.spatial_dims = 3
        self.conv1 = nn.Conv3d(in_channels, base, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(base)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(base, base, blocks=2, stride=1)
        self.layer2 = self._make_layer(base, base * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base * 2, base * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base * 4, base * 8, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.fc = nn.Linear(base * 8, num_classes)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock3D(in_channels, out_channels, stride=stride)]
        for _ in range(1, int(blocks)):
            layers.append(BasicBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"ResNet3D expects [B,C,D,H,W], got {tuple(x.shape)}.")
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(self.dropout(x))


def build_resnet18_3d(
    num_classes: int = 2,
    n_input_channels: int = 1,
    dropout: float = 0.0,
    base_channels: int = 32,
    **kwargs,
) -> ResNet3D:
    """Build the research and inference model from explicit architecture settings."""
    del kwargs
    return ResNet3D(
        in_channels=int(n_input_channels),
        num_classes=int(num_classes),
        dropout=float(dropout),
        base_channels=int(base_channels),
    )


def load_model_from_bundle(
    weights_path,
    model_config: dict,
    device: str | torch.device = "cpu",
) -> nn.Module:
    model = build_resnet18_3d(
        num_classes=int(model_config.get("num_classes", 2)),
        n_input_channels=int(model_config.get("n_input_channels", 1)),
        dropout=float(model_config.get("dropout", 0.0)),
        base_channels=int(model_config.get("base_channels", 32)),
    )
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Packaged model weights do not contain a state dictionary.")
    cleaned = {str(k).removeprefix("module."): v for k, v in checkpoint.items()}
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model

