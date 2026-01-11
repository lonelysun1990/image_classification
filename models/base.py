"""Base model class for all classifiers."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn


class BaseClassifier(nn.Module, ABC):
    """
    Abstract base class for all classifier models.
    
    All custom models should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        pass
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get the number of parameters in the model.
        
        Args:
            trainable_only: If True, only count trainable parameters
        
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration as a dictionary.
        Override in subclasses to include model-specific config.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            'model_class': self.__class__.__name__,
            'num_parameters': self.get_num_parameters(),
            'trainable_parameters': self.get_num_parameters(trainable_only=True),
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default configuration for this model class.
        Override in subclasses to provide default hyperparameters.
        
        Returns:
            Dictionary with default configuration
        """
        return {}
