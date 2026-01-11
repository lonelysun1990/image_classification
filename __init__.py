"""
Image Classification Package

A modular, production-ready package for image classification experiments.
Supports multiple model architectures, data augmentation, class imbalance handling,
hyperparameter tuning, and experiment tracking with Weights & Biases.
"""

__version__ = "0.1.0"

from .data import (
    create_dataloaders,
    get_transforms,
    ImbalancedDatasetSampler,
)
from .models import (
    get_model,
    list_models,
    register_model,
    CNNClassifier,
    VisionTransformer,
)
from .training import Trainer, evaluate_model
from .experiments import HyperparameterTuner
from .visualization import plot_training_curves, plot_confusion_matrices

__all__ = [
    # Data
    "create_dataloaders",
    "get_transforms",
    "ImbalancedDatasetSampler",
    # Models
    "get_model",
    "list_models",
    "register_model",
    "CNNClassifier",
    "VisionTransformer",
    # Training
    "Trainer",
    "evaluate_model",
    # Experiments
    "HyperparameterTuner",
    # Visualization
    "plot_training_curves",
    "plot_confusion_matrices",
]
