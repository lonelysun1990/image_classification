"""Plotting utilities for visualization."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn


def plot_training_curves(
    results: List[Tuple[Dict[str, List[float]], str]],
    figsize: Tuple[int, int] = (15, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training curves for multiple models.
    
    Args:
        results: List of tuples (history_dict, model_name)
                 where history_dict has keys: 'train_loss', 'train_acc', 'val_acc'
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for i, (history, name) in enumerate(results):
        marker = markers[i % len(markers)]
        
        # Training Loss
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label=name, marker=marker, markersize=4)
        
        # Training Accuracy
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label=name, marker=marker, markersize=4)
        
        # Validation Accuracy
        if 'val_acc' in history:
            axes[2].plot(history['val_acc'], label=name, marker=marker, markersize=4)
    
    # Configure axes
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Validation Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrices(
    results: List[Tuple[np.ndarray, np.ndarray, str]],
    class_names: Optional[List[str]] = None,
    figsize_per_plot: Tuple[int, int] = (7, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot confusion matrices for multiple models.
    
    Args:
        results: List of tuples (y_true, y_pred, model_name)
        class_names: List of class names for axis labels
        figsize_per_plot: Size per confusion matrix subplot
        save_path: Path to save the figure (optional)
    """
    n_models = len(results)
    fig_width = figsize_per_plot[0] * n_models
    fig_height = figsize_per_plot[1]
    
    fig, axes = plt.subplots(1, n_models, figsize=(fig_width, fig_height))
    
    # Handle single model case
    if n_models == 1:
        axes = [axes]
    
    for ax, (y_true, y_pred, model_name) in zip(axes, results):
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap='Blues')
        
        n_classes = cm.shape[0]
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        
        if class_names is not None:
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model_name} Confusion Matrix')
        
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_sample_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    n_samples: int = 16,
    n_cols: int = 4,
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot sample predictions from the model.
    
    Args:
        model: Trained model
        dataloader: DataLoader with samples
        device: Device to run inference
        class_names: List of class names
        n_samples: Number of samples to display
        n_cols: Number of columns in the grid
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    model.eval()
    
    # Get samples
    images_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            
            images_list.append(images.cpu())
            labels_list.extend(labels.numpy())
            preds_list.extend(preds.cpu().numpy())
            
            if len(labels_list) >= n_samples:
                break
    
    images = torch.cat(images_list)[:n_samples]
    labels = labels_list[:n_samples]
    preds = preds_list[:n_samples]
    
    # Plot
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (img, true_label, pred_label) in enumerate(zip(images, labels, preds)):
        ax = axes[i]
        
        # Handle different image formats
        if img.shape[0] == 1:
            ax.imshow(img.squeeze().numpy(), cmap='gray')
        else:
            ax.imshow(img.permute(1, 2, 0).numpy())
        
        # Color based on correct/incorrect
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(
            f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}',
            color=color,
            fontsize=8,
        )
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def print_model_summary(
    models_info: List[Tuple[nn.Module, Dict[str, List[float]], str]],
) -> None:
    """
    Print a summary table for multiple models.
    
    Args:
        models_info: List of tuples (model, history, model_name)
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Parameters':>15} {'Final Val Acc':>15} {'Best Val Acc':>15}")
    print("-" * 70)
    
    for model, history, name in models_info:
        params = sum(p.numel() for p in model.parameters())
        final_acc = history['val_acc'][-1] if 'val_acc' in history else 0
        best_acc = max(history['val_acc']) if 'val_acc' in history else 0
        print(f"{name:<20} {params:>15,} {final_acc:>14.2f}% {best_acc:>14.2f}%")
    
    print("=" * 70)


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the distribution of classes in a dataset.
    
    Args:
        labels: Array of labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(unique)), counts, color='steelblue')
    
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            str(count),
            ha='center',
            fontsize=8,
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
