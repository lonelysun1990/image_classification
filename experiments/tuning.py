"""
Hyperparameter tuning utilities.

Provides grid search and random search for hyperparameter optimization
with optional wandb sweep integration.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import itertools
import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..models import get_model
from ..training import Trainer


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    config: Dict[str, Any]
    val_acc: float
    val_loss: float
    train_acc: float
    train_loss: float
    history: Dict[str, List[float]] = field(default_factory=dict)
    model_path: Optional[str] = None


class HyperparameterTuner:
    """
    Hyperparameter tuning class supporting grid search and random search.
    
    Args:
        model_name: Name of the model in the registry
        device: Device to train on
        base_model_config: Base configuration for the model
        
    Example:
        >>> tuner = HyperparameterTuner("cnn", device)
        >>> param_grid = {
        ...     'lr': [1e-3, 1e-4],
        ...     'dropout': [0.2, 0.3, 0.4],
        ... }
        >>> results = tuner.grid_search(train_loader, val_loader, param_grid)
    """
    
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        base_model_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.base_model_config = base_model_config or {}
        self.results: List[ExperimentResult] = []
    
    def grid_search(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        param_grid: Dict[str, List[Any]],
        num_epochs: int = 10,
        use_wandb: bool = False,
        wandb_project: str = "hyperparameter-search",
        verbose: bool = True,
    ) -> List[ExperimentResult]:
        """
        Perform grid search over hyperparameters.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            param_grid: Dictionary mapping parameter names to lists of values
            num_epochs: Number of epochs per trial
            use_wandb: Whether to log to W&B
            wandb_project: W&B project name
            verbose: Whether to print progress
        
        Returns:
            List of ExperimentResult objects sorted by validation accuracy
        """
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        if verbose:
            print(f"Grid search: {len(combinations)} configurations to try")
        
        for i, combo in enumerate(combinations):
            config = dict(zip(keys, combo))
            
            if verbose:
                print(f"\n[{i + 1}/{len(combinations)}] Config: {config}")
            
            result = self._run_trial(
                train_loader, val_loader,
                config, num_epochs,
                use_wandb, wandb_project,
                verbose,
            )
            self.results.append(result)
        
        # Sort by validation accuracy
        self.results.sort(key=lambda x: x.val_acc, reverse=True)
        
        if verbose:
            self._print_summary()
        
        return self.results
    
    def random_search(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        param_distributions: Dict[str, Any],
        n_trials: int = 10,
        num_epochs: int = 10,
        use_wandb: bool = False,
        wandb_project: str = "hyperparameter-search",
        verbose: bool = True,
        seed: int = 42,
    ) -> List[ExperimentResult]:
        """
        Perform random search over hyperparameters.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            param_distributions: Dict mapping parameter names to:
                - List of values (uniform random choice)
                - Tuple (min, max) for uniform float
                - Tuple (min, max, 'log') for log-uniform float
                - Tuple (min, max, 'int') for uniform integer
            n_trials: Number of random configurations to try
            num_epochs: Number of epochs per trial
            use_wandb: Whether to log to W&B
            wandb_project: W&B project name
            verbose: Whether to print progress
            seed: Random seed
        
        Returns:
            List of ExperimentResult objects sorted by validation accuracy
        """
        random.seed(seed)
        
        if verbose:
            print(f"Random search: {n_trials} trials")
        
        for i in range(n_trials):
            config = self._sample_config(param_distributions)
            
            if verbose:
                print(f"\n[{i + 1}/{n_trials}] Config: {config}")
            
            result = self._run_trial(
                train_loader, val_loader,
                config, num_epochs,
                use_wandb, wandb_project,
                verbose,
            )
            self.results.append(result)
        
        # Sort by validation accuracy
        self.results.sort(key=lambda x: x.val_acc, reverse=True)
        
        if verbose:
            self._print_summary()
        
        return self.results
    
    def _sample_config(self, param_distributions: Dict[str, Any]) -> Dict[str, Any]:
        """Sample a configuration from parameter distributions."""
        config = {}
        
        for key, dist in param_distributions.items():
            if isinstance(dist, list):
                # Uniform choice from list
                config[key] = random.choice(dist)
            elif isinstance(dist, tuple):
                if len(dist) == 2:
                    # Uniform float
                    config[key] = random.uniform(dist[0], dist[1])
                elif len(dist) == 3:
                    if dist[2] == 'log':
                        # Log-uniform float
                        import math
                        log_min = math.log(dist[0])
                        log_max = math.log(dist[1])
                        config[key] = math.exp(random.uniform(log_min, log_max))
                    elif dist[2] == 'int':
                        # Uniform integer
                        config[key] = random.randint(dist[0], dist[1])
                    else:
                        raise ValueError(f"Unknown distribution type: {dist[2]}")
            else:
                raise ValueError(f"Unknown distribution format for {key}: {dist}")
        
        return config
    
    def _run_trial(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        num_epochs: int,
        use_wandb: bool,
        wandb_project: str,
        verbose: bool,
    ) -> ExperimentResult:
        """Run a single trial with the given configuration."""
        # Separate model config from training config
        model_config = self.base_model_config.copy()
        training_config = {}
        
        model_param_names = {'in_channels', 'num_classes', 'base_filters', 'num_conv_layers',
                           'fc_hidden_dim', 'dropout', 'use_batch_norm', 'img_size',
                           'patch_size', 'embed_dim', 'num_heads', 'num_layers', 'mlp_ratio'}
        
        for key, value in config.items():
            if key in model_param_names:
                model_config[key] = value
            else:
                training_config[key] = value
        
        # Create model
        model = get_model(self.model_name, **model_config)
        
        # Get learning rate from config or use default
        lr = training_config.get('lr', 1e-3)
        
        # Create trainer
        trainer = Trainer(model, self.device)
        
        # Train
        history = trainer.fit(
            train_loader, val_loader,
            num_epochs=num_epochs,
            lr=lr,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_config=config,
            verbose=verbose,
        )
        
        # Get final metrics
        result = ExperimentResult(
            config=config,
            val_acc=history['val_acc'][-1],
            val_loss=history['val_loss'][-1],
            train_acc=history['train_acc'][-1],
            train_loss=history['train_loss'][-1],
            history=history,
        )
        
        return result
    
    def _print_summary(self):
        """Print summary of search results."""
        print("\n" + "=" * 70)
        print("HYPERPARAMETER SEARCH RESULTS")
        print("=" * 70)
        print(f"{'Rank':<6} {'Val Acc':<12} {'Train Acc':<12} {'Config'}")
        print("-" * 70)
        
        for i, result in enumerate(self.results[:10]):  # Top 10
            config_str = ', '.join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in result.config.items())
            print(f"{i + 1:<6} {result.val_acc:<12.2f} {result.train_acc:<12.2f} {config_str}")
        
        print("=" * 70)
        print(f"Best config: {self.results[0].config}")
        print(f"Best val accuracy: {self.results[0].val_acc:.2f}%")
        print("=" * 70)
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get the best configuration found."""
        if not self.results:
            raise ValueError("No results yet. Run grid_search or random_search first.")
        return self.results[0].config
    
    def get_best_result(self) -> ExperimentResult:
        """Get the best result found."""
        if not self.results:
            raise ValueError("No results yet. Run grid_search or random_search first.")
        return self.results[0]


# Convenience aliases
GridSearch = HyperparameterTuner
RandomSearch = HyperparameterTuner
