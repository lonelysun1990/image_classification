"""
Samplers for handling class imbalance.

Provides various strategies for dealing with imbalanced datasets:
- Oversampling minority classes
- Undersampling majority classes
- Weighted sampling
"""

from typing import Callable, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler


class ImbalancedDatasetSampler(Sampler):
    """
    Sampler for imbalanced datasets that ensures each class is sampled
    with equal probability during training.
    
    This effectively oversamples minority classes and undersamples majority classes.
    
    Args:
        dataset: The dataset to sample from
        labels: Optional pre-computed labels. If None, will try to extract from dataset
        callback_get_label: Optional function to extract label from dataset item
        num_samples: Number of samples to draw per epoch. If None, uses len(dataset)
    
    Example:
        >>> sampler = ImbalancedDatasetSampler(train_dataset)
        >>> train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=32)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        labels: Optional[List[int]] = None,
        callback_get_label: Optional[Callable] = None,
        num_samples: Optional[int] = None,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        
        # Get labels
        if labels is not None:
            self.labels = np.array(labels)
        elif callback_get_label is not None:
            self.labels = np.array([callback_get_label(dataset, i) for i in range(len(dataset))])
        else:
            self.labels = self._get_labels_from_dataset()
        
        # Compute class weights
        self.class_counts = np.bincount(self.labels)
        self.class_weights = 1.0 / self.class_counts
        
        # Assign weight to each sample
        self.sample_weights = torch.DoubleTensor(
            [self.class_weights[label] for label in self.labels]
        )
    
    def _get_labels_from_dataset(self) -> np.ndarray:
        """Try to extract labels from the dataset."""
        # Try common dataset attributes
        if hasattr(self.dataset, 'targets'):
            return np.array(self.dataset.targets)
        elif hasattr(self.dataset, 'labels'):
            return np.array(self.dataset.labels)
        elif hasattr(self.dataset, 'y'):
            return np.array(self.dataset.y)
        else:
            # Fall back to iterating through dataset
            labels = []
            for i in range(len(self.dataset)):
                _, label = self.dataset[i]
                labels.append(label)
            return np.array(labels)
    
    def __iter__(self):
        """Generate indices for sampling."""
        return iter(
            torch.multinomial(
                self.sample_weights,
                self.num_samples,
                replacement=True
            ).tolist()
        )
    
    def __len__(self) -> int:
        return self.num_samples
    
    def get_class_distribution(self) -> dict:
        """Return the class distribution in the dataset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


def get_weighted_sampler(
    labels: List[int],
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for imbalanced datasets.
    
    This is an alternative to ImbalancedDatasetSampler using PyTorch's
    built-in WeightedRandomSampler.
    
    Args:
        labels: List of labels for all samples
        num_samples: Number of samples to draw. If None, uses len(labels)
    
    Returns:
        WeightedRandomSampler instance
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = torch.DoubleTensor([class_weights[label] for label in labels])
    
    num_samples = num_samples or len(labels)
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


def create_class_balanced_subset(
    dataset: Dataset,
    samples_per_class: int,
    labels: Optional[List[int]] = None,
) -> torch.utils.data.Subset:
    """
    Create a class-balanced subset by sampling equal numbers from each class.
    
    This is useful for creating smaller balanced validation sets.
    
    Args:
        dataset: The dataset to sample from
        samples_per_class: Number of samples to take from each class
        labels: Optional pre-computed labels
    
    Returns:
        Subset of the original dataset with balanced classes
    """
    if labels is None:
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
    else:
        labels = np.array(labels)
    
    unique_classes = np.unique(labels)
    indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) < samples_per_class:
            # If not enough samples, take all with replacement
            selected = np.random.choice(
                cls_indices, samples_per_class, replace=True
            )
        else:
            selected = np.random.choice(
                cls_indices, samples_per_class, replace=False
            )
        indices.extend(selected.tolist())
    
    np.random.shuffle(indices)
    return torch.utils.data.Subset(dataset, indices)
