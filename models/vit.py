"""Vision Transformer (ViT) model."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from .base import BaseClassifier
from .registry import register_model


class PatchEmbedding(nn.Module):
    """
    Convert image into patch embeddings.
    
    Splits image into non-overlapping patches and projects each patch
    to an embedding vector.
    
    Args:
        img_size: Input image size (assumes square images)
        patch_size: Size of each patch (assumes square patches)
        in_channels: Number of input channels
        embed_dim: Dimension of patch embeddings
    """
    
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, height, width)
        # Output: (batch, num_patches, embed_dim)
        x = self.projection(x)  # (batch, embed_dim, h/patch, w/patch)
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


@register_model("vit")
class VisionTransformer(BaseClassifier):
    """
    Vision Transformer for image classification.
    
    Implementation based on "An Image is Worth 16x16 Words: Transformers for
    Image Recognition at Scale" (Dosovitskiy et al., 2020).
    
    Args:
        img_size: Input image size (assumes square images)
        patch_size: Size of each patch
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Dimension of patch/token embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        mlp_ratio: Ratio of MLP hidden dim to embed_dim
        dropout: Dropout rate
        attn_dropout: Attention dropout rate
    """
    
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        
        # Store config
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.attn_dropout_rate = attn_dropout
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Take class token output
        x = self.norm(x[:, 0])
        
        # Classification head
        x = self.head(x)
        
        return x
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'mlp_ratio': self.mlp_ratio,
            'dropout': self.dropout_rate,
            'attn_dropout': self.attn_dropout_rate,
        })
        return config
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            'img_size': 28,
            'patch_size': 7,
            'in_channels': 1,
            'num_classes': 10,
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 4,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'attn_dropout': 0.0,
        }


@register_model("vit_tiny")
class VisionTransformerTiny(VisionTransformer):
    """Tiny ViT variant with fewer parameters."""
    
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=dropout,
        )


@register_model("vit_small")
class VisionTransformerSmall(VisionTransformer):
    """Small ViT variant."""
    
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=192,
            num_heads=6,
            num_layers=6,
            mlp_ratio=4.0,
            dropout=dropout,
        )
