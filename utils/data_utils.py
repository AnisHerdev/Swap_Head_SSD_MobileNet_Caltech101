import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import json


class ImageClassificationDataset(Dataset):
    """
    Custom dataset for image classification
    """
    def __init__(self, data_dir: str, transform=None, split='train'):
        """
        Args:
            data_dir: Directory containing the dataset
            transform: Optional transform to be applied on a sample
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Load class mapping
        self.class_to_idx = self._load_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Load file paths and labels
        self.samples = self._load_samples()
    
    def _load_class_mapping(self) -> Dict[str, int]:
        """Load class mapping from JSON file or create default"""
        mapping_file = os.path.join(self.data_dir, 'class_mapping.json')
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
        else:
            # Create default mapping based on subdirectories
            # Check if we have a nested structure (train/val/test folders)
            if os.path.exists(os.path.join(self.data_dir, 'train')):
                # Nested structure: data_dir/train/class1, data_dir/val/class1, etc.
                train_dir = os.path.join(self.data_dir, 'train')
                classes = sorted([d for d in os.listdir(train_dir) 
                                if os.path.isdir(os.path.join(train_dir, d))])
            else:
                # Flat structure: data_dir/class1, data_dir/class2, etc.
                classes = sorted([d for d in os.listdir(self.data_dir) 
                                if os.path.isdir(os.path.join(self.data_dir, d))])
            return {cls: idx for idx, cls in enumerate(classes)}
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their corresponding labels"""
        samples = []
        
        # Check if we have a nested structure (train/val/test folders)
        if os.path.exists(os.path.join(self.data_dir, 'train')):
            # Nested structure: data_dir/train/class1, data_dir/val/class1, etc.
            split_dir = os.path.join(self.data_dir, self.split)
            for class_name, class_idx in self.class_to_idx.items():
                class_dir = os.path.join(split_dir, class_name)
                if os.path.exists(class_dir):
                    for filename in os.listdir(class_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_dir, filename)
                            samples.append((img_path, class_idx))
        else:
            # Flat structure: data_dir/class1, data_dir/class2, etc.
            for class_name, class_idx in self.class_to_idx.items():
                class_dir = os.path.join(self.data_dir, class_name)
                if os.path.exists(class_dir):
                    for filename in os.listdir(class_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_dir, filename)
                            samples.append((img_path, class_idx))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(input_size: int = 320, is_training: bool = True) -> transforms.Compose:
    """
    Get transforms for data preprocessing
    Args:
        input_size: Size to resize the image to
        is_training: Whether this is for training (includes augmentation)
    Returns:
        transforms: Composition of transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4,
                       input_size: int = 320) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        input_size: Input size for the model
    Returns:
        train_loader, val_loader, test_loader: Data loaders
    """
    # Create datasets
    train_dataset = ImageClassificationDataset(
        data_dir, 
        transform=get_transforms(input_size, is_training=True),
        split='train'
    )
    
    val_dataset = ImageClassificationDataset(
        data_dir, 
        transform=get_transforms(input_size, is_training=False),
        split='val'
    )
    
    test_dataset = ImageClassificationDataset(
        data_dir, 
        transform=get_transforms(input_size, is_training=False),
        split='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_names(data_dir: str) -> List[str]:
    """
    Get class names from the dataset directory
    Args:
        data_dir: Directory containing the dataset
    Returns:
        class_names: List of class names
    """
    # Check if we have a nested structure (train/val/test folders)
    if os.path.exists(os.path.join(data_dir, 'train')):
        # Nested structure: data_dir/train/class1, data_dir/val/class1, etc.
        train_dir = os.path.join(data_dir, 'train')
        classes = sorted([d for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))])
    else:
        # Flat structure: data_dir/class1, data_dir/class2, etc.
        classes = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    return classes


def save_class_mapping(data_dir: str, output_file: str = 'class_mapping.json'):
    """
    Save class mapping to a JSON file
    Args:
        data_dir: Directory containing the dataset
        output_file: Output file path for the mapping
    """
    classes = get_class_names(data_dir)
    class_mapping = {cls: idx for idx, cls in enumerate(classes)}
    
    with open(output_file, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"Class mapping saved to {output_file}")
    print(f"Found {len(classes)} classes: {classes}")


def calculate_dataset_stats(data_dir: str) -> Dict[str, Any]:
    """
    Calculate dataset statistics
    Args:
        data_dir: Directory containing the dataset
    Returns:
        stats: Dictionary containing dataset statistics
    """
    class_names = get_class_names(data_dir)
    class_counts = {}
    total_images = 0
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count
            total_images += count
    
    stats = {
        'total_images': total_images,
        'num_classes': len(class_names),
        'class_counts': class_counts,
        'class_names': class_names
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/your/dataset"
    
    # Save class mapping
    save_class_mapping(data_dir)
    
    # Calculate and print dataset statistics
    stats = calculate_dataset_stats(data_dir)
    print(f"Dataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Class counts: {stats['class_counts']}") 