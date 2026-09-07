"""Model registry for the original 2-D experiments and the DaT 3-D model."""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
from torch import nn

# Type for model builder functions: takes (num_classes, **kwargs) and returns nn.Module
ModelBuilder = Callable[..., nn.Module]


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
    """The behavior-preserving 3-D ResNet18-style DaT classifier."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.0, base_channels: int = 32):
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
        self.layer1 = self._make_layer(base, base, 2, 1)
        self.layer2 = self._make_layer(base, base * 2, 2, 2)
        self.layer3 = self._make_layer(base * 2, base * 4, 2, 2)
        self.layer4 = self._make_layer(base * 4, base * 8, 2, 2)
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


def build_resnet18_3d(num_classes: int = 2, n_input_channels: int = 1, dropout: float = 0.0, base_channels: int = 32, **kwargs) -> ResNet3D:
    del kwargs
    return ResNet3D(
        in_channels=int(n_input_channels), num_classes=int(num_classes),
        dropout=float(dropout), base_channels=int(base_channels),
    )


def load_model_from_bundle(weights_path, model_config: dict, device: str | torch.device = "cpu") -> nn.Module:
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
    model.load_state_dict({str(k).removeprefix("module."): v for k, v in checkpoint.items()}, strict=True)
    model.to(device)
    model.eval()
    return model


def _build_resnet(resnet_factory: Callable[..., nn.Module], num_classes: int, input_size: int = 224, **kwargs) -> nn.Module:
    model = resnet_factory(weights=None)
    if input_size <= 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _resnet18_builder(num_classes: int, input_size: int = 32, **kwargs) -> nn.Module:
    """Build ResNet18 with stem selected by input size."""
    from torchvision.models import resnet18

    return _build_resnet(resnet18, num_classes=num_classes, input_size=input_size, **kwargs)


def _resnet18_3d_builder(
    num_classes: int,
    input_size=None,
    n_input_channels: int = 1,
    dropout: float = 0.0,
    base_channels: int = 32,
    **kwargs,
) -> nn.Module:
    """Build the dimension-aware DaT model without changing torchvision ResNet18."""
    del input_size
    return build_resnet18_3d(
        num_classes=num_classes,
        n_input_channels=n_input_channels,
        dropout=dropout,
        base_channels=base_channels,
        **kwargs,
    )


# Registry of available models
# Key: model name, Value: {"builder": builder_fn, "default_input_size": expected_size}
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "builder": _resnet18_builder,
        "default_input_size": 32,
    },
    "resnet18_3d": {
        "builder": _resnet18_3d_builder,
        "default_input_size": 96,
        "spatial_dims": 3,
        "input_channels": 1,
        "num_classes": 2,
    },
}


def _get_model_info(model_name: str) -> Dict[str, Any]:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def get_model(model_name: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Get a model builder by name and instantiate it.
    
    Args:
        model_name: Name of the model (``resnet18`` or ``resnet18_3d``)
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to the model builder
    
    Returns:
        Instantiated PyTorch model
    
    Raises:
        ValueError: If model_name is not registered
    """
    model_info = _get_model_info(model_name)
    builder = model_info["builder"]
    return builder(num_classes=num_classes, **kwargs)


def get_available_models() -> list:
    """Return a list of available model names."""
    return list(MODEL_REGISTRY.keys())


def register_model(name: str, builder: ModelBuilder, default_input_size: int = 224) -> None:
    """
    Register a new model in the registry.
    
    Args:
        name: Name for the model
        builder: Function that takes (num_classes, **kwargs) and returns nn.Module
        default_input_size: Expected input image size for this model
    """
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is already registered")
    
    MODEL_REGISTRY[name] = {
        "builder": builder,
        "default_input_size": default_input_size,
    }
