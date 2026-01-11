"""
Data augmentation transforms.

Best Practice: Data augmentation should ONLY be applied to training data.
Test/validation data should only have normalization applied to ensure
consistent evaluation metrics.
"""

from typing import Optional, Tuple
import torch
from torchvision import transforms


def get_transforms(
    img_size: int = 28,
    mean: Tuple[float, ...] = (0.5,),
    std: Tuple[float, ...] = (0.5,),
    train: bool = True,
    augmentation_level: str = "standard",
) -> transforms.Compose:
    """
    Get transforms for training or evaluation.
    
    Args:
        img_size: Target image size
        mean: Normalization mean (per channel)
        std: Normalization std (per channel)
        train: If True, include data augmentation; if False, only normalize
        augmentation_level: One of 'none', 'light', 'standard', 'heavy'
    
    Returns:
        A torchvision transforms.Compose object
    """
    if train:
        return TrainTransform(img_size, mean, std, augmentation_level)
    else:
        return TestTransform(img_size, mean, std)


class TrainTransform:
    """
    Training transforms with configurable augmentation levels.
    
    Augmentation levels:
    - 'none': Only resize and normalize (useful for debugging)
    - 'light': Random horizontal flip only
    - 'standard': Flip + small rotation + small affine transforms
    - 'heavy': Standard + color jitter + random erasing
    """
    
    def __init__(
        self,
        img_size: int = 28,
        mean: Tuple[float, ...] = (0.5,),
        std: Tuple[float, ...] = (0.5,),
        augmentation_level: str = "standard",
    ):
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.augmentation_level = augmentation_level
        self.transform = self._build_transform()
    
    def _build_transform(self) -> transforms.Compose:
        transform_list = []
        
        # Resize if needed
        transform_list.append(transforms.Resize((self.img_size, self.img_size)))
        
        if self.augmentation_level == "none":
            pass  # No augmentation
        
        elif self.augmentation_level == "light":
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        elif self.augmentation_level == "standard":
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                ),
            ])
        
        elif self.augmentation_level == "heavy":
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.15, 0.15),
                    scale=(0.85, 1.15),
                    shear=10,
                ),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                ),
            ])
        
        else:
            raise ValueError(
                f"Unknown augmentation_level: {self.augmentation_level}. "
                "Choose from: 'none', 'light', 'standard', 'heavy'"
            )
        
        # Convert to tensor and normalize (always applied)
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        # Random erasing for heavy augmentation (applied after ToTensor)
        if self.augmentation_level == "heavy":
            transform_list.append(
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
            )
        
        return transforms.Compose(transform_list)
    
    def __call__(self, img):
        return self.transform(img)


class TestTransform:
    """
    Test/validation transforms - only resize and normalize.
    No augmentation to ensure consistent evaluation.
    """
    
    def __init__(
        self,
        img_size: int = 28,
        mean: Tuple[float, ...] = (0.5,),
        std: Tuple[float, ...] = (0.5,),
    ):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    def __call__(self, img):
        return self.transform(img)
