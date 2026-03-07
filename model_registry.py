"""
Model registry for easy addition of new models.
Each model is registered with metadata about its expected input and number of classes.
"""
import torch.nn as nn
from torchvision.models import (
    densenet121,
    efficientnet_b0,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    resnet50,
    vgg16_bn,
)
from typing import Callable, Dict, Any

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
}


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
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_info = MODEL_REGISTRY[model_name]
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
