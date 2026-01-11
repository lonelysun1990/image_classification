"""Tests for training module."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, '..')

from image_classification.training import Trainer, evaluate_model, compute_metrics
from image_classification.models import CNNClassifier


def create_dummy_dataloaders(batch_size=16, train_size=100, val_size=20):
    """Create dummy dataloaders for testing."""
    train_data = torch.randn(train_size, 1, 28, 28)
    train_labels = torch.randint(0, 10, (train_size,))
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_data = torch.randn(val_size, 1, 28, 28)
    val_labels = torch.randint(0, 10, (val_size,))
    val_dataset = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


class TestTrainer:
    """Tests for Trainer class."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = CNNClassifier(in_channels=1, num_classes=10)
        device = torch.device('cpu')
        
        trainer = Trainer(model, device)
        
        assert trainer.model is model
        assert trainer.device == device
    
    def test_trainer_fit_basic(self):
        """Test basic training loop."""
        model = CNNClassifier(in_channels=1, num_classes=10)
        device = torch.device('cpu')
        trainer = Trainer(model, device)
        
        train_loader, val_loader = create_dummy_dataloaders()
        
        history = trainer.fit(
            train_loader, val_loader,
            num_epochs=2,
            lr=1e-3,
            use_wandb=False,
            verbose=False,
        )
        
        assert 'train_loss' in history
        assert 'train_acc' in history
        assert 'val_loss' in history
        assert 'val_acc' in history
        assert len(history['train_loss']) == 2
    
    def test_trainer_evaluate(self):
        """Test evaluation method."""
        model = CNNClassifier(in_channels=1, num_classes=10)
        device = torch.device('cpu')
        trainer = Trainer(model, device)
        
        _, val_loader = create_dummy_dataloaders()
        
        # Run forward pass to build model
        _ = model(torch.randn(1, 1, 28, 28))
        
        loss, acc = trainer.evaluate(val_loader)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
    
    def test_trainer_history_tracking(self):
        """Test that history is properly tracked."""
        model = CNNClassifier(in_channels=1, num_classes=10)
        device = torch.device('cpu')
        trainer = Trainer(model, device)
        
        train_loader, val_loader = create_dummy_dataloaders()
        
        history = trainer.fit(
            train_loader, val_loader,
            num_epochs=3,
            use_wandb=False,
            verbose=False,
        )
        
        # All history lists should have same length
        assert len(history['train_loss']) == 3
        assert len(history['train_acc']) == 3
        assert len(history['val_loss']) == 3
        assert len(history['val_acc']) == 3
        assert len(history['lr']) == 3


class TestMetrics:
    """Tests for metrics functions."""
    
    def test_compute_metrics(self):
        """Test compute_metrics function."""
        import numpy as np
        
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 2])  # One mistake
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Accuracy should be 5/6
        assert abs(metrics['accuracy'] - 5/6) < 0.01
    
    def test_evaluate_model(self):
        """Test evaluate_model function."""
        model = CNNClassifier(in_channels=1, num_classes=10)
        device = torch.device('cpu')
        
        _, val_loader = create_dummy_dataloaders()
        
        # Run forward pass to build model
        _ = model(torch.randn(1, 1, 28, 28))
        
        results = evaluate_model(model, val_loader, device, print_report=False)
        
        assert 'metrics' in results
        assert 'confusion_matrix' in results
        assert 'y_true' in results
        assert 'y_pred' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
