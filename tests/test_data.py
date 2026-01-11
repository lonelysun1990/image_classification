"""Tests for data module."""

import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset

import sys
sys.path.insert(0, '..')

from image_classification.data.transforms import get_transforms, TrainTransform, TestTransform
from image_classification.data.samplers import ImbalancedDatasetSampler, get_weighted_sampler
from image_classification.data.utils import train_val_test_split, get_class_weights


class TestTransforms:
    """Tests for data transforms."""
    
    def test_train_transform_output_shape(self):
        """Test that train transform produces correct output shape."""
        transform = get_transforms(img_size=28, train=True)
        
        # Create a dummy PIL image
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8))
        
        output = transform(img)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 28, 28)
    
    def test_test_transform_output_shape(self):
        """Test that test transform produces correct output shape."""
        transform = get_transforms(img_size=28, train=False)
        
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8))
        
        output = transform(img)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 28, 28)
    
    def test_augmentation_levels(self):
        """Test different augmentation levels."""
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8))
        
        for level in ['none', 'light', 'standard', 'heavy']:
            transform = TrainTransform(augmentation_level=level)
            output = transform(img)
            assert output.shape == (1, 28, 28)
    
    def test_invalid_augmentation_level(self):
        """Test that invalid augmentation level raises error."""
        with pytest.raises(ValueError):
            TrainTransform(augmentation_level='invalid')


class TestSamplers:
    """Tests for data samplers."""
    
    def test_imbalanced_sampler_length(self):
        """Test that ImbalancedDatasetSampler returns correct length."""
        # Create imbalanced dataset
        labels = [0] * 100 + [1] * 10  # 10:1 imbalance
        data = torch.randn(110, 3, 28, 28)
        dataset = TensorDataset(data, torch.tensor(labels))
        
        sampler = ImbalancedDatasetSampler(dataset, labels=labels)
        
        assert len(sampler) == len(dataset)
    
    def test_imbalanced_sampler_iteration(self):
        """Test that ImbalancedDatasetSampler can be iterated."""
        labels = [0] * 100 + [1] * 10
        data = torch.randn(110, 3, 28, 28)
        dataset = TensorDataset(data, torch.tensor(labels))
        
        sampler = ImbalancedDatasetSampler(dataset, labels=labels)
        indices = list(sampler)
        
        assert len(indices) == len(dataset)
        assert all(0 <= idx < len(dataset) for idx in indices)
    
    def test_weighted_sampler(self):
        """Test weighted sampler creation."""
        labels = [0, 0, 0, 1, 1, 2]
        sampler = get_weighted_sampler(labels)
        
        assert len(sampler) == len(labels)


class TestUtils:
    """Tests for data utilities."""
    
    def test_train_val_test_split_sizes(self):
        """Test that split produces correct sizes."""
        data = torch.randn(100, 3, 28, 28)
        labels = torch.randint(0, 10, (100,))
        dataset = TensorDataset(data, labels)
        
        train, val, test = train_val_test_split(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify=False,
        )
        
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
    
    def test_train_val_test_split_stratified(self):
        """Test stratified split maintains class distribution."""
        # Create dataset with known class distribution
        labels = [0] * 50 + [1] * 30 + [2] * 20
        data = torch.randn(100, 3, 28, 28)
        dataset = TensorDataset(data, torch.tensor(labels))
        
        train, val, test = train_val_test_split(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify=True,
        )
        
        # Check total size
        assert len(train) + len(val) + len(test) == 100
    
    def test_get_class_weights(self):
        """Test class weight computation."""
        labels = [0, 0, 0, 1, 1, 2]  # Imbalanced
        data = torch.randn(6, 3, 28, 28)
        dataset = TensorDataset(data, torch.tensor(labels))
        
        weights = get_class_weights(dataset, labels=labels)
        
        assert len(weights) == 3
        # Minority class should have higher weight
        assert weights[2] > weights[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
