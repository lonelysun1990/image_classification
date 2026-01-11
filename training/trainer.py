"""
Trainer class for model training with wandb integration.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """
    Trainer class for training classification models.
    
    Supports:
    - Training with validation
    - Early stopping
    - Learning rate scheduling
    - Weights & Biases logging
    - Model checkpointing
    
    Args:
        model: The model to train
        device: Device to train on
        optimizer: Optimizer instance (if None, Adam is used)
        scheduler: Learning rate scheduler (if None, CosineAnnealingLR is used)
        criterion: Loss function (if None, CrossEntropyLoss is used)
    
    Example:
        >>> trainer = Trainer(model, device)
        >>> history = trainer.fit(train_loader, val_loader, num_epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        lr: float = 1e-3,
        use_wandb: bool = True,
        wandb_project: str = "image-classification",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        early_stopping_patience: Optional[int] = None,
        save_best_model: bool = False,
        model_save_path: str = "best_model.pt",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            lr: Learning rate (used if optimizer not provided)
            use_wandb: Whether to log to Weights & Biases
            wandb_project: W&B project name
            wandb_run_name: W&B run name
            wandb_config: Additional config to log to W&B
            early_stopping_patience: Stop if val acc doesn't improve for N epochs
            save_best_model: Whether to save the best model
            model_save_path: Path to save the best model
            verbose: Whether to print progress
        
        Returns:
            Training history dictionary
        """
        # Initialize optimizer if not provided
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Initialize scheduler if not provided
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs
            )
        
        # Initialize wandb
        if use_wandb and WANDB_AVAILABLE:
            config = {
                'epochs': num_epochs,
                'learning_rate': lr,
                'optimizer': self.optimizer.__class__.__name__,
                'scheduler': self.scheduler.__class__.__name__ if self.scheduler else None,
                'device': str(self.device),
            }
            if wandb_config:
                config.update(wandb_config)
            if hasattr(self.model, 'get_config'):
                config.update(self.model.get_config())
            
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=config,
            )
            wandb.watch(self.model, log="all", log_freq=100)
        
        # Early stopping
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self._validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': current_lr,
                })
            
            # Check for best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                if save_best_model:
                    self._save_checkpoint(model_save_path)
            else:
                patience_counter += 1
            
            # Print progress
            if verbose:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Finish wandb run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        return self.history
    
    def _train_epoch(self, dataloader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, dataloader: DataLoader) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def evaluate(self, dataloader: DataLoader) -> tuple:
        """Evaluate the model on a dataset."""
        return self._validate(dataloader)
