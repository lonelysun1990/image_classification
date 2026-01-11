"""CNN-based classification models."""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseClassifier
from .registry import register_model


@register_model("cnn")
class CNNClassifier(BaseClassifier):
    """
    Convolutional Neural Network for image classification.
    
    A flexible CNN architecture with configurable number of layers,
    filters, and fully connected layers.
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
        base_filters: Number of filters in the first conv layer
        num_conv_layers: Number of convolutional layers
        fc_hidden_dim: Hidden dimension for fully connected layers
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        base_filters: int = 32,
        num_conv_layers: int = 3,
        fc_hidden_dim: int = 128,
        dropout: float = 0.25,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        # Store config
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.num_conv_layers = num_conv_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build convolutional layers
        conv_layers = []
        in_ch = in_channels
        out_ch = base_filters
        
        for i in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            ])
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.extend([
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout),
            ])
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)  # Cap at 256 filters
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate flattened size (will be set in forward pass)
        self._flat_size = None
        
        # Placeholder for FC layers (will be built on first forward)
        self.fc_layers = None
        self._fc_built = False
    
    def _build_fc_layers(self, flat_size: int):
        """Build fully connected layers based on flattened conv output size."""
        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, self.fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_hidden_dim, self.num_classes),
        )
        self._flat_size = flat_size
        self._fc_built = True
        
        # Move to same device as conv layers
        device = next(self.conv_layers.parameters()).device
        self.fc_layers = self.fc_layers.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Build FC layers on first forward pass
        if not self._fc_built:
            self._build_fc_layers(x.size(1))
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'base_filters': self.base_filters,
            'num_conv_layers': self.num_conv_layers,
            'fc_hidden_dim': self.fc_hidden_dim,
            'dropout': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
        })
        return config
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            'in_channels': 1,
            'num_classes': 10,
            'base_filters': 32,
            'num_conv_layers': 3,
            'fc_hidden_dim': 128,
            'dropout': 0.25,
            'use_batch_norm': True,
        }


@register_model("cnn_small")
class CNNClassifierSmall(CNNClassifier):
    """Smaller CNN variant with fewer parameters."""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=16,
            num_conv_layers=2,
            fc_hidden_dim=64,
            dropout=dropout,
            use_batch_norm=True,
        )


@register_model("cnn_large")
class CNNClassifierLarge(CNNClassifier):
    """Larger CNN variant with more parameters."""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=64,
            num_conv_layers=4,
            fc_hidden_dim=256,
            dropout=dropout,
            use_batch_norm=True,
        )
