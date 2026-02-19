"""
Model registry for easy addition of new models.
Each model is registered with metadata about its expected input and number of classes.
"""
import torch.nn as nn
from torchvision.models import *
from typing import Callable, Dict, Any

# Type for model builder functions: takes (num_classes, **kwargs) and returns nn.Module
ModelBuilder = Callable[[int], nn.Module]


def _resnet18_builder(num_classes: int, input_size: int = 32, dropout: float = 0.0, **kwargs) -> nn.Module:
    """Build ResNet18 adapted for small image sizes (like CIFAR)."""
    model = resnet18(weights=None)
    
    # Adapt for small image inputs (e.g., 32x32)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # Replace classifier for the target number of classes
    if dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


# Registry of available models
# Key: model name, Value: {"builder": builder_fn, "default_input_size": expected_size}
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "builder": _resnet18_builder,
        "default_input_size": 32,
    },
}


def get_model(model_name: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Get a model builder by name and instantiate it.
    
    Args:
        model_name: Name of the model (e.g., 'resnet18', 'vgg16')
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to the model builder (e.g., dropout)
    
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
