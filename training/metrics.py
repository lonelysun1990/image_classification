"""Evaluation metrics for classification models."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
    
    Returns:
        Tuple of (true_labels, predicted_labels, predicted_probabilities)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = 'weighted',
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names for the report
        average: Averaging method for multi-class metrics
    
    Returns:
        Dictionary with computed metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    return metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    print_report: bool = True,
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        class_names: Optional list of class names
        print_report: Whether to print classification report
    
    Returns:
        Dictionary with metrics, predictions, and confusion matrix
    """
    y_true, y_pred, y_prob = get_predictions(model, dataloader, device)
    
    metrics = compute_metrics(y_true, y_pred, class_names)
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }
    
    if print_report:
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0,
        ))
        print("=" * 60)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted F1 Score: {metrics['f1']:.4f}")
        print("=" * 60)
    
    return results


def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute per-class accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Array of per-class accuracies
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return per_class_acc
