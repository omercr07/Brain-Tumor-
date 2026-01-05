"""
Data Loading and Preprocessing Module
Handles dataset loading, transformations, and data splitting
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class BrainTumorDataset(Dataset):
    """
    Custom Dataset class for Brain Tumor images
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of labels (0 for 'no', 1 for 'yes')
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a single image and its label
        
        Args:
            idx: Index of the item
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size=224, augment=True):
    """
    Get data transformation pipelines
    
    Args:
        image_size: Target image size (default: 224)
        augment: Whether to apply data augmentation for training
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def load_dataset(data_dir, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load dataset from directory structure
    
    Expected structure:
    data_dir/
        yes/
            *.jpg, *.jpeg, *.png
        no/
            *.jpg, *.jpeg, *.png
    
    Args:
        data_dir: Root directory of the dataset
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (train_paths, train_labels, val_paths, val_labels, 
                test_paths, test_labels)
    """
    image_paths = []
    labels = []
    
    # Supported image extensions
    extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    # Load 'yes' class (label=1)
    yes_dir = os.path.join(data_dir, 'yes')
    if os.path.exists(yes_dir):
        for filename in os.listdir(yes_dir):
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(yes_dir, filename))
                labels.append(1)
    
    # Also check brain_tumor_dataset/yes
    brain_yes_dir = os.path.join(data_dir, 'brain_tumor_dataset', 'yes')
    if os.path.exists(brain_yes_dir):
        for filename in os.listdir(brain_yes_dir):
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(brain_yes_dir, filename))
                labels.append(1)
    
    # Load 'no' class (label=0)
    no_dir = os.path.join(data_dir, 'no')
    if os.path.exists(no_dir):
        for filename in os.listdir(no_dir):
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(no_dir, filename))
                labels.append(0)
    
    # Also check brain_tumor_dataset/no
    brain_no_dir = os.path.join(data_dir, 'brain_tumor_dataset', 'no')
    if os.path.exists(brain_no_dir):
        for filename in os.listdir(brain_no_dir):
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(brain_no_dir, filename))
                labels.append(0)
    
    print(f"Total images found: {len(image_paths)}")
    print(f"  - Class 'yes' (tumor): {labels.count(1)}")
    print(f"  - Class 'no' (no tumor): {labels.count(0)}")
    
    # Split into train and test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, 
        random_state=random_state, stratify=labels
    )
    
    # Split train into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_size,
        random_state=random_state, stratify=train_labels
    )
    
    print(f"\nDataset split:")
    print(f"  - Training: {len(train_paths)} images")
    print(f"  - Validation: {len(val_paths)} images")
    print(f"  - Test: {len(test_paths)} images")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def get_dataloaders(data_dir, batch_size=32, image_size=224, 
                    num_workers=4, augment=True):
    """
    Create DataLoader objects for training, validation, and testing
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size for DataLoader
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        augment: Whether to apply data augmentation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load dataset
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        load_dataset(data_dir)
    
    # Get transforms
    train_transform, val_transform = get_transforms(image_size, augment)
    
    # Create datasets
    train_dataset = BrainTumorDataset(train_paths, train_labels, train_transform)
    val_dataset = BrainTumorDataset(val_paths, val_labels, val_transform)
    test_dataset = BrainTumorDataset(test_paths, test_labels, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    data_dir = "dataset"
    print("Testing data loading...")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size=8, num_workers=0
    )
    
    # Test a batch
    for images, labels in train_loader:
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Label shape: {labels.shape}")
        break

