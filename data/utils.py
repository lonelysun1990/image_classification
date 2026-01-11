"""Data utilities for splitting and preprocessing."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split


def train_val_test_split(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_seed: Optional[int] = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        dataset: The dataset to split
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        stratify: If True, maintain class distribution in each split
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    
    Example:
        >>> train_set, val_set, test_set = train_val_test_split(
        ...     dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        ... )
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    n_samples = len(dataset)
    
    if stratify:
        return _stratified_split(dataset, train_ratio, val_ratio, test_ratio)
    else:
        # Simple random split
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        test_size = n_samples - train_size - val_size
        
        return random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed) if random_seed else None,
        )


def _stratified_split(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[Subset, Subset, Subset]:
    """Perform stratified split maintaining class distribution."""
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    unique_classes = np.unique(labels)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        
        n_cls = len(cls_indices)
        n_train = int(n_cls * train_ratio)
        n_val = int(n_cls * val_ratio)
        
        train_indices.extend(cls_indices[:n_train].tolist())
        val_indices.extend(cls_indices[n_train:n_train + n_val].tolist())
        test_indices.extend(cls_indices[n_train + n_val:].tolist())
    
    # Shuffle each split
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def get_class_weights(
    dataset: Dataset,
    labels: Optional[List[int]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies.
    
    Useful for weighted loss functions like nn.CrossEntropyLoss(weight=...).
    
    Args:
        dataset: The dataset to compute weights for
        labels: Optional pre-computed labels
        normalize: If True, normalize weights to sum to num_classes
    
    Returns:
        Tensor of class weights
    """
    if labels is None:
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
    else:
        labels = np.array(labels)
    
    class_counts = np.bincount(labels)
    total = len(labels)
    num_classes = len(class_counts)
    
    # Inverse frequency weighting
    weights = total / (num_classes * class_counts)
    
    if normalize:
        weights = weights * num_classes / weights.sum()
    
    return torch.FloatTensor(weights)


def compute_dataset_statistics(
    dataset: Dataset,
    num_samples: Optional[int] = None,
) -> Dict[str, Tuple[float, ...]]:
    """
    Compute mean and std of a dataset for normalization.
    
    Args:
        dataset: The dataset to compute statistics for
        num_samples: Number of samples to use. If None, use entire dataset
    
    Returns:
        Dict with 'mean' and 'std' tuples
    """
    if num_samples is None:
        num_samples = len(dataset)
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Accumulate statistics
    pixel_sum = None
    pixel_sq_sum = None
    num_pixels = 0
    
    for idx in indices:
        img, _ = dataset[idx]
        if not isinstance(img, torch.Tensor):
            from torchvision import transforms
            img = transforms.ToTensor()(img)
        
        if pixel_sum is None:
            num_channels = img.shape[0]
            pixel_sum = torch.zeros(num_channels)
            pixel_sq_sum = torch.zeros(num_channels)
        
        pixel_sum += img.sum(dim=(1, 2))
        pixel_sq_sum += (img ** 2).sum(dim=(1, 2))
        num_pixels += img.shape[1] * img.shape[2]
    
    mean = pixel_sum / num_pixels
    std = torch.sqrt(pixel_sq_sum / num_pixels - mean ** 2)
    
    return {
        'mean': tuple(mean.tolist()),
        'std': tuple(std.tolist()),
    }
