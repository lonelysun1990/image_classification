# Image Classification Package

A modular, production-ready Python package for image classification experiments.

## Features

- **Multiple Model Architectures**: CNN and Vision Transformer (ViT) with easy model registration
- **Data Processing**: Train/val/test splitting, data augmentation, class imbalance handling
- **Experiment Tracking**: Built-in Weights & Biases integration
- **Hyperparameter Tuning**: Grid search and random search utilities
- **Visualization**: Training curves, confusion matrices, sample predictions

## Installation

```bash
# Install from source
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from image_classification import (
    create_dataloaders,
    get_model,
    Trainer,
    plot_training_curves,
)

# Setup device
device = torch.device('mps' if torch.backends.mps.is_available() 
                      else 'cuda' if torch.cuda.is_available() 
                      else 'cpu')

# Create dataloaders
dataloaders, info = create_dataloaders(
    dataset_name="fashion_mnist",
    batch_size=64,
    augmentation_level="standard",
)

# Create model
model = get_model("cnn", in_channels=1, num_classes=10)

# Train
trainer = Trainer(model, device)
history = trainer.fit(
    dataloaders['train'],
    dataloaders['val'],
    num_epochs=10,
    use_wandb=True,
)

# Visualize
plot_training_curves([(history, 'CNN')])
```

## Available Models

| Model | Name | Description |
|-------|------|-------------|
| CNN | `cnn` | Standard CNN with configurable layers |
| CNN Small | `cnn_small` | Smaller variant with fewer parameters |
| CNN Large | `cnn_large` | Larger variant with more parameters |
| ViT | `vit` | Vision Transformer |
| ViT Tiny | `vit_tiny` | Smaller ViT variant |
| ViT Small | `vit_small` | Small ViT variant |

## Adding Custom Models

```python
from image_classification.models import register_model, BaseClassifier

@register_model("my_model")
class MyModel(BaseClassifier):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # ... your model implementation
    
    def forward(self, x):
        # ... forward pass
        return x
```

## Data Augmentation

Augmentation is only applied to training data. Levels available:
- `none`: No augmentation
- `light`: Random horizontal flip
- `standard`: Flip + rotation + affine transforms
- `heavy`: Standard + color jitter + random erasing

## Hyperparameter Tuning

```python
from image_classification import HyperparameterTuner

tuner = HyperparameterTuner("cnn", device)

# Grid search
results = tuner.grid_search(
    train_loader, val_loader,
    param_grid={
        'lr': [1e-3, 1e-4],
        'dropout': [0.2, 0.3],
    },
    num_epochs=10,
)

# Random search
results = tuner.random_search(
    train_loader, val_loader,
    param_distributions={
        'lr': (1e-5, 1e-2, 'log'),  # Log-uniform
        'dropout': (0.1, 0.5),  # Uniform
    },
    n_trials=20,
)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=image_classification
```

## Project Structure

```
image_classification/
├── __init__.py
├── data/
│   ├── datasets.py      # Dataset classes
│   ├── transforms.py    # Data augmentation
│   ├── samplers.py      # Class imbalance handling
│   └── utils.py         # Train/val/test splitting
├── models/
│   ├── base.py          # Base classifier class
│   ├── cnn.py           # CNN models
│   ├── vit.py           # Vision Transformer
│   └── registry.py      # Model registry
├── training/
│   ├── trainer.py       # Training loop with wandb
│   └── metrics.py       # Evaluation metrics
├── experiments/
│   └── tuning.py        # Hyperparameter search
├── visualization/
│   └── plots.py         # Plotting utilities
├── tests/               # Unit tests
├── configs/             # Configuration files
└── demo.ipynb           # Demo notebook
```

## License

MIT License
