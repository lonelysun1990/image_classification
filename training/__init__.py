"""Training module with trainer and metrics."""

from .trainer import Trainer
from .metrics import (
    evaluate_model,
    compute_metrics,
    get_predictions,
)

__all__ = [
    "Trainer",
    "evaluate_model",
    "compute_metrics",
    "get_predictions",
]
