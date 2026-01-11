"""Dataset classes and data loading utilities."""

from typing import Dict, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from .transforms import get_transforms
from .samplers import ImbalancedDatasetSampler
from .utils import train_val_test_split


# FashionMNIST class names
FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


class FashionMNISTDataset:
    """
    Wrapper for FashionMNIST dataset with train/val/test splits.
    
    Args:
        data_dir: Directory to download/load data
        train_ratio: Proportion of training data for train split
        val_ratio: Proportion of training data for validation split
        augmentation_level: Level of data augmentation ('none', 'light', 'standard', 'heavy')
        random_seed: Random seed for reproducibility
    """
    
    CLASS_NAMES = FASHION_MNIST_CLASSES
    NUM_CLASSES = 10
    IMG_SIZE = 28
    IN_CHANNELS = 1
    
    def __init__(
        self,
        data_dir: str = "./data",
        train_ratio: float = 0.85,
        val_ratio: float = 0.15,
        augmentation_level: str = "standard",
        random_seed: int = 42,
    ):
        self.data_dir = data_dir
        self.augmentation_level = augmentation_level
        
        # Load raw datasets
        train_data = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
        )
        test_data = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
        )
        
        # Split training data into train/val
        train_subset, val_subset, _ = train_val_test_split(
            train_data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=0.0,  # We'll use the official test set
            stratify=True,
            random_seed=random_seed,
        )
        
        # Create dataset wrappers with appropriate transforms
        self.train_dataset = TransformDataset(
            train_subset,
            transform=get_transforms(train=True, augmentation_level=augmentation_level),
        )
        self.val_dataset = TransformDataset(
            val_subset,
            transform=get_transforms(train=False),
        )
        self.test_dataset = TransformDataset(
            test_data,
            transform=get_transforms(train=False),
        )
    
    def get_dataloaders(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        use_imbalanced_sampler: bool = False,
    ) -> Dict[str, DataLoader]:
        """
        Get DataLoaders for train, val, and test sets.
        
        Args:
            batch_size: Batch size for all loaders
            num_workers: Number of workers for data loading
            use_imbalanced_sampler: Use ImbalancedDatasetSampler for training
        
        Returns:
            Dict with 'train', 'val', 'test' DataLoaders
        """
        train_sampler = None
        train_shuffle = True
        
        if use_imbalanced_sampler:
            train_sampler = ImbalancedDatasetSampler(self.train_dataset)
            train_shuffle = False  # Sampler handles this
        
        return {
            'train': DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=train_shuffle,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
            ),
            'val': DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
            'test': DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        }


class TransformDataset(Dataset):
    """
    Wrapper dataset that applies a transform to another dataset.
    
    Useful for applying different transforms to train/val/test splits
    that come from the same base dataset.
    """
    
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Handle Subset datasets
        if isinstance(self.dataset, Subset):
            img, label = self.dataset.dataset[self.dataset.indices[idx]]
        else:
            img, label = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    @property
    def targets(self):
        """Return targets for compatibility with samplers."""
        if isinstance(self.dataset, Subset):
            base_targets = self.dataset.dataset.targets
            return [base_targets[i] for i in self.dataset.indices]
        return self.dataset.targets


def create_dataloaders(
    dataset_name: str = "fashion_mnist",
    data_dir: str = "./data",
    batch_size: int = 64,
    train_ratio: float = 0.85,
    val_ratio: float = 0.15,
    augmentation_level: str = "standard",
    use_imbalanced_sampler: bool = False,
    num_workers: int = 0,
    random_seed: int = 42,
) -> Tuple[Dict[str, DataLoader], Dict]:
    """
    Create dataloaders for a dataset.
    
    Args:
        dataset_name: Name of the dataset ('fashion_mnist' supported)
        data_dir: Directory for data storage
        batch_size: Batch size
        train_ratio: Ratio for training split
        val_ratio: Ratio for validation split
        augmentation_level: Data augmentation level
        use_imbalanced_sampler: Whether to use imbalanced sampler
        num_workers: Number of data loading workers
        random_seed: Random seed
    
    Returns:
        Tuple of (dataloaders_dict, dataset_info_dict)
    """
    if dataset_name.lower() == "fashion_mnist":
        dataset = FashionMNISTDataset(
            data_dir=data_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            augmentation_level=augmentation_level,
            random_seed=random_seed,
        )
        
        dataloaders = dataset.get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            use_imbalanced_sampler=use_imbalanced_sampler,
        )
        
        info = {
            'num_classes': dataset.NUM_CLASSES,
            'class_names': dataset.CLASS_NAMES,
            'img_size': dataset.IMG_SIZE,
            'in_channels': dataset.IN_CHANNELS,
            'train_size': len(dataset.train_dataset),
            'val_size': len(dataset.val_dataset),
            'test_size': len(dataset.test_dataset),
        }
        
        return dataloaders, info
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
