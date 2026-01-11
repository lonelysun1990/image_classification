"""
Model registry for easy model creation and management.

This module provides a centralized way to register, retrieve, and list
available models. New models can be easily added using the @register_model
decorator.
"""

from typing import Any, Callable, Dict, List, Optional, Type
import torch.nn as nn


# Global model registry
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model class in the registry.
    
    Args:
        name: Name to register the model under (case-insensitive)
    
    Returns:
        Decorator function
    
    Example:
        @register_model("my_model")
        class MyModel(BaseClassifier):
            def __init__(self, ...):
                ...
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_model(
    name: str,
    **kwargs,
) -> nn.Module:
    """
    Get a model instance by name.
    
    Args:
        name: Name of the model (case-insensitive)
        **kwargs: Arguments to pass to the model constructor
    
    Returns:
        Instantiated model
    
    Raises:
        ValueError: If model name is not found in registry
    
    Example:
        >>> model = get_model("cnn", in_channels=1, num_classes=10)
        >>> model = get_model("vit", img_size=28, patch_size=7)
    """
    name = name.lower()
    
    if name not in MODEL_REGISTRY:
        available = list_models()
        raise ValueError(
            f"Model '{name}' not found. Available models: {available}"
        )
    
    model_cls = MODEL_REGISTRY[name]
    return model_cls(**kwargs)


def list_models() -> List[str]:
    """
    List all registered model names.
    
    Returns:
        List of registered model names
    """
    return list(MODEL_REGISTRY.keys())


def get_model_config(name: str) -> Dict[str, Any]:
    """
    Get the default configuration for a model.
    
    Args:
        name: Name of the model (case-insensitive)
    
    Returns:
        Dictionary with default configuration
    """
    name = name.lower()
    
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found.")
    
    model_cls = MODEL_REGISTRY[name]
    
    if hasattr(model_cls, 'get_default_config'):
        return model_cls.get_default_config()
    
    return {}


def get_model_class(name: str) -> Type[nn.Module]:
    """
    Get the model class by name (without instantiating).
    
    Args:
        name: Name of the model (case-insensitive)
    
    Returns:
        Model class
    """
    name = name.lower()
    
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found.")
    
    return MODEL_REGISTRY[name]
