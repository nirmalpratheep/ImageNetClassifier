import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageNetDataset(Dataset):
    """
    Custom Dataset class for ImageNet data.
    
    Expected structure:
    data/
    ├── train/
    │   ├── 0/    (class 0 images) OR n01440764/
    │   ├── 1/    (class 1 images) OR n01443537/
    │   └── ...
    └── val/
        ├── 0/    (class 0 images) OR n01440764/
        ├── 1/    (class 1 images) OR n01443537/
        └── ...
    """
    
    def __init__(self, data_dir, transform=None, subset_size=None, max_samples_per_class=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all image paths and labels
        self.samples = self._load_samples(subset_size, max_samples_per_class)
        
        # Determine number of classes from the data
        if self.samples:
            max_class = max(sample[1] for sample in self.samples)
            self.num_classes = max_class + 1
        else:
            self.num_classes = 0
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        print(f"Number of classes: {self.num_classes}")
    
    def _load_samples(self, subset_size=None, max_samples_per_class=None):
        """Load all image paths and their corresponding labels."""
        samples = []
        
        # Get all class folders (could be numeric 0-999 or ImageNet IDs like n01440764)
        class_folders = [f for f in os.listdir(self.data_dir) 
                        if os.path.isdir(os.path.join(self.data_dir, f))]
        
        # Try to sort: if numeric, sort numerically; otherwise sort alphabetically
        try:
            # Check if folders are numeric
            sorted_folders = sorted(class_folders, key=lambda x: int(x) if x.isdigit() else float('inf'))
            class_folders = sorted_folders
        except:
            # If not numeric, just sort alphabetically
            class_folders.sort()
        
        # Limit classes if subset_size is specified
        if subset_size:
            class_folders = class_folders[:subset_size]
        
        # Create mapping from folder name to class index
        class_to_idx = {}
        for idx, class_folder in enumerate(class_folders):
            class_to_idx[class_folder] = idx
        
        for class_folder in class_folders:
            class_path = os.path.join(self.data_dir, class_folder)
            class_idx = class_to_idx[class_folder]
            
            # Get all image files in the class folder
            image_files = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit samples per class if specified
            if max_samples_per_class and len(image_files) > max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                samples.append((image_path, class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size=224, augmentation=True):
    """Get data transforms for training and validation."""
    
    if augmentation:
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Training transforms without augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_dir, batch_size=32, image_size=224, 
                       num_workers=4, subset_size=None, augmentation=True,
                       max_samples_per_class=None):
    """
    Create training and validation data loaders.
    
    Args:
        data_dir (str): Base directory containing train and val folders
        batch_size (int): Batch size for data loaders
        image_size (int): Target image size
        num_workers (int): Number of worker processes for data loading
        subset_size (int): If specified, only use first N classes
        augmentation (bool): Whether to apply data augmentation
        max_samples_per_class (int): Maximum number of samples per class to load
    
    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)
    """
    
    # Get transforms
    train_transform, val_transform = get_transforms(image_size, augmentation)
    
    # Create datasets
    train_dataset = ImageNetDataset(
        data_dir=os.path.join(data_dir, 'train'),
        transform=train_transform,
        subset_size=subset_size,
        max_samples_per_class=max_samples_per_class
    )
    
    val_dataset = ImageNetDataset(
        data_dir=os.path.join(data_dir, 'val'),
        transform=val_transform,
        subset_size=subset_size,
        max_samples_per_class=max_samples_per_class
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset, val_dataset
