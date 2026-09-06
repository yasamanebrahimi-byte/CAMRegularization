"""
Model registry for easy addition of new models.
Each model is registered with metadata about its expected input and number of classes.
"""
import torch.nn as nn
from torchvision.models import (
    convnext_tiny,
    densenet121,
    efficientnet_b0,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    resnet50,
    swin_t,
    vit_b_16,
    vgg16_bn,
)
from typing import Callable, Dict, Any

from dat_model import build_resnet18_3d

# Type for model builder functions: takes (num_classes, **kwargs) and returns nn.Module
ModelBuilder = Callable[..., nn.Module]


def _build_resnet(resnet_factory: Callable[..., nn.Module], num_classes: int, input_size: int = 224, **kwargs) -> nn.Module:
    model = resnet_factory(weights=None)
    if input_size <= 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _resnet18_builder(num_classes: int, input_size: int = 32, **kwargs) -> nn.Module:
    """Build ResNet18 with stem selected by input size."""
    return _build_resnet(resnet18, num_classes=num_classes, input_size=input_size, **kwargs)


def _resnet34_builder(num_classes: int, input_size: int = 32, **kwargs) -> nn.Module:
    return _build_resnet(resnet34, num_classes=num_classes, input_size=input_size, **kwargs)


def _resnet50_builder(num_classes: int, input_size: int = 32, **kwargs) -> nn.Module:
    return _build_resnet(resnet50, num_classes=num_classes, input_size=input_size, **kwargs)


def _vgg16_bn_builder(num_classes: int, **kwargs) -> nn.Module:
    model = vgg16_bn(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _densenet121_builder(num_classes: int, **kwargs) -> nn.Module:
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def _mobilenet_v3_small_builder(num_classes: int, **kwargs) -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _mobilenet_v3_large_builder(num_classes: int, **kwargs) -> nn.Module:
    model = mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _efficientnet_b0_builder(num_classes: int, **kwargs) -> nn.Module:
    model = efficientnet_b0(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _vit_b_16_builder(num_classes: int, input_size: int = 224, **kwargs) -> nn.Module:
    model = vit_b_16(weights=None, image_size=input_size)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def _convnext_tiny_builder(num_classes: int, **kwargs) -> nn.Module:
    model = convnext_tiny(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _swin_t_builder(num_classes: int, **kwargs) -> nn.Module:
    model = swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


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
    "resnet34": {
        "builder": _resnet34_builder,
        "default_input_size": 32,
    },
    "resnet50": {
        "builder": _resnet50_builder,
        "default_input_size": 32,
    },
    "vgg16_bn": {
        "builder": _vgg16_bn_builder,
        "default_input_size": 224,
    },
    "densenet121": {
        "builder": _densenet121_builder,
        "default_input_size": 224,
    },
    "mobilenet_v3_small": {
        "builder": _mobilenet_v3_small_builder,
        "default_input_size": 224,
    },
    "mobilenet_v3_large": {
        "builder": _mobilenet_v3_large_builder,
        "default_input_size": 224,
    },
    "efficientnet_b0": {
        "builder": _efficientnet_b0_builder,
        "default_input_size": 224,
    },
    "vit_b_16": {
        "builder": _vit_b_16_builder,
        "default_input_size": 224,
    },
    "convnext_tiny": {
        "builder": _convnext_tiny_builder,
        "default_input_size": 224,
    },
    "swin_t": {
        "builder": _swin_t_builder,
        "default_input_size": 224,
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
        model_name: Name of the model (e.g., 'resnet18', 'vgg16')
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
