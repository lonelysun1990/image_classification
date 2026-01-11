"""Tests for models module."""

import pytest
import torch

import sys
sys.path.insert(0, '..')

from image_classification.models import (
    get_model,
    list_models,
    register_model,
    CNNClassifier,
    VisionTransformer,
)
from image_classification.models.base import BaseClassifier


class TestModelRegistry:
    """Tests for model registry."""
    
    def test_list_models(self):
        """Test that list_models returns registered models."""
        models = list_models()
        
        assert isinstance(models, list)
        assert 'cnn' in models
        assert 'vit' in models
    
    def test_get_model_cnn(self):
        """Test getting CNN model."""
        model = get_model('cnn', in_channels=1, num_classes=10)
        
        assert isinstance(model, CNNClassifier)
        assert isinstance(model, BaseClassifier)
    
    def test_get_model_vit(self):
        """Test getting ViT model."""
        model = get_model('vit', img_size=28, patch_size=7, in_channels=1, num_classes=10)
        
        assert isinstance(model, VisionTransformer)
        assert isinstance(model, BaseClassifier)
    
    def test_get_model_invalid(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            get_model('invalid_model')
    
    def test_register_custom_model(self):
        """Test registering a custom model."""
        @register_model('test_model')
        class TestModel(BaseClassifier):
            def __init__(self, num_classes=10):
                super().__init__()
                self.fc = torch.nn.Linear(100, num_classes)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))
        
        assert 'test_model' in list_models()
        model = get_model('test_model')
        assert isinstance(model, TestModel)


class TestCNNClassifier:
    """Tests for CNN classifier."""
    
    def test_forward_pass(self):
        """Test CNN forward pass."""
        model = CNNClassifier(in_channels=1, num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_different_input_channels(self):
        """Test CNN with different input channels."""
        for in_channels in [1, 3]:
            model = CNNClassifier(in_channels=in_channels, num_classes=10)
            x = torch.randn(2, in_channels, 28, 28)
            
            output = model(x)
            assert output.shape == (2, 10)
    
    def test_different_num_classes(self):
        """Test CNN with different number of classes."""
        for num_classes in [2, 10, 100]:
            model = CNNClassifier(in_channels=1, num_classes=num_classes)
            x = torch.randn(2, 1, 28, 28)
            
            output = model(x)
            assert output.shape == (2, num_classes)
    
    def test_get_config(self):
        """Test getting model config."""
        model = CNNClassifier(in_channels=1, num_classes=10, dropout=0.3)
        config = model.get_config()
        
        assert 'model_class' in config
        assert config['in_channels'] == 1
        assert config['num_classes'] == 10
        assert config['dropout'] == 0.3
    
    def test_num_parameters(self):
        """Test counting parameters."""
        model = CNNClassifier(in_channels=1, num_classes=10)
        # Run forward to build FC layers
        _ = model(torch.randn(1, 1, 28, 28))
        
        num_params = model.get_num_parameters()
        assert num_params > 0


class TestVisionTransformer:
    """Tests for Vision Transformer."""
    
    def test_forward_pass(self):
        """Test ViT forward pass."""
        model = VisionTransformer(
            img_size=28, patch_size=7, in_channels=1, num_classes=10
        )
        x = torch.randn(2, 1, 28, 28)
        
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_different_patch_sizes(self):
        """Test ViT with different patch sizes."""
        for patch_size in [4, 7, 14]:
            model = VisionTransformer(
                img_size=28, patch_size=patch_size, in_channels=1, num_classes=10
            )
            x = torch.randn(2, 1, 28, 28)
            
            output = model(x)
            assert output.shape == (2, 10)
    
    def test_different_embed_dims(self):
        """Test ViT with different embedding dimensions."""
        for embed_dim in [64, 128, 256]:
            model = VisionTransformer(
                img_size=28, patch_size=7, in_channels=1, num_classes=10,
                embed_dim=embed_dim
            )
            x = torch.randn(2, 1, 28, 28)
            
            output = model(x)
            assert output.shape == (2, 10)
    
    def test_get_config(self):
        """Test getting model config."""
        model = VisionTransformer(
            img_size=28, patch_size=7, in_channels=1, num_classes=10,
            embed_dim=128, num_heads=4
        )
        config = model.get_config()
        
        assert config['img_size'] == 28
        assert config['patch_size'] == 7
        assert config['embed_dim'] == 128
        assert config['num_heads'] == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
