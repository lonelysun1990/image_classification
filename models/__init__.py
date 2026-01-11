"""Models module with registry for easy model addition."""

from .base import BaseClassifier
from .cnn import CNNClassifier
from .vit import VisionTransformer, PatchEmbedding
from .registry import (
    register_model,
    get_model,
    list_models,
    MODEL_REGISTRY,
)

__all__ = [
    "BaseClassifier",
    "CNNClassifier",
    "VisionTransformer",
    "PatchEmbedding",
    "register_model",
    "get_model",
    "list_models",
    "MODEL_REGISTRY",
]
