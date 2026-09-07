"""
Model registry for easy addition of new models.
Each model is registered with metadata about its expected input and number of classes.
"""
import torch.nn as nn
from torchvision.models import resnet18
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
