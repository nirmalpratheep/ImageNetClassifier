import torch
from torchvision import datasets, transforms
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, CoarseDropout,
    Normalize, ColorJitter, PadIfNeeded, RandomCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import io
import os
import random
from torch.utils.data import Dataset, Subset
from typing import List, Tuple

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


# ImageNet-1K statistics (RGB)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _coarse_dropout_fill_value_from_mean(mean_rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    """Convert mean RGB (0–1) to 0–255 scale for CoarseDropout fill color."""
    return tuple(int(m * 255.0) for m in mean_rgb)


class AlbumentationsAdapter:
    """Adapter to make Albumentations transforms compatible with torchvision datasets."""
    def __init__(self, transform: Compose):
        self.transform = transform

    def __call__(self, img):
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented["image"]


class HuggingFaceImageNetDataset:
    """Custom dataset class for Hugging Face ImageNet-1K with streaming support."""
    
    def __init__(self, dataset, transform=None, max_samples=None):
        self.dataset = dataset
        self.transform = transform
        self.max_samples = max_samples
        self._current_samples = 0
    
    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.dataset))
        return len(self.dataset)
    
    def __iter__(self):
        """Iterator for streaming datasets."""
        for i, item in enumerate(self.dataset):
            if self.max_samples and i >= self.max_samples:
                break
            
            # Load image
            if 'image' in item:
                image = item['image']
            else:
                # Handle different image formats
                image = Image.open(io.BytesIO(item['bytes']))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Get label
            label = item.get('label', 0)
            
            yield image, label
    
    def __getitem__(self, idx):
        """Get item by index (for non-streaming datasets)."""
        if hasattr(self.dataset, '__getitem__'):
            item = self.dataset[idx]
            
            # Load image
            if 'image' in item:
                image = item['image']
            else:
                image = Image.open(io.BytesIO(item['bytes']))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Get label
            label = item.get('label', 0)
            
            return image, label
        else:
            raise NotImplementedError("Indexing not supported for streaming datasets")


class ImageNetSubsetDataset(Dataset):
    """Custom dataset for ImageNet subset with numeric class folders (e.g., 00500, 00501, etc.)"""
    
    def __init__(self, root_dir: str, transform=None, class_mapping: dict = None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_mapping = class_mapping or {}
        
        # Scan directory for class folders and images
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        self._scan_directory()
    
    def _scan_directory(self):
        """Scan the directory structure and build sample list."""
        class_dirs = [d for d in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, d))]
        
        # Sort class directories to ensure consistent ordering
        class_dirs.sort()
        
        # Map class directories to indices
        self.classes = class_dirs
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_dirs)}
        
        # Collect all image files
        for class_dir in class_dirs:
            class_path = os.path.join(self.root_dir, class_dir)
            class_idx = self.class_to_idx[class_dir]
            
            # Get all image files in this class directory
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                self.samples.append((image_path, class_idx))
        
        print(f"Found {len(self.classes)} classes with {len(self.samples)} total samples")
        print(f"Classes: {self.classes[:5]}{'...' if len(self.classes) > 5 else ''}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_train_val_split(dataset: Dataset, val_ratio: float = 0.2, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """Create train/validation split from a single dataset."""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # Set seed for reproducible splits
    random.seed(seed)
    random.shuffle(indices)
    
    # Calculate split point
    val_size = int(dataset_size * val_ratio)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    return train_dataset, val_dataset


def detect_data_structure(data_dir: str) -> str:
    """Detect the structure of the data directory."""
    if not os.path.exists(data_dir):
        return "none"
    
    # Check for traditional ImageNet structure (train/val directories)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        return "imagenet_traditional"
    
    # Check for numeric class directories (ImageNet subset)
    subdirs = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    # Check if directories are numeric (like 00500, 00501, etc.)
    numeric_dirs = [d for d in subdirs if d.isdigit()]
    
    if len(numeric_dirs) >= 5:  # At least 5 numeric class directories
        return "imagenet_subset"
    
    # Check for any class-like directories
    if len(subdirs) >= 2:
        return "generic_classes"
    
    return "unknown"


def get_transforms():
    """Return transforms for ImageNet-1K dataset."""
    # ImageNet-1K transforms
    train_transforms = Compose([
        Resize(height=256, width=256, p=1.0),
        RandomCrop(height=224, width=224, p=1.0),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    test_transforms = Compose([
        Resize(height=256, width=256, p=1.0),
        RandomCrop(height=224, width=224, p=1.0),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    return AlbumentationsAdapter(train_transforms), AlbumentationsAdapter(test_transforms)


def get_datasets(streaming: bool = True, max_samples: int = None, data_dir: str = "./data", val_ratio: float = 0.2):
    """Return ImageNet-1K train/test datasets with appropriate transforms."""
    train_transforms, test_transforms = get_transforms()

    # Detect data structure
    data_structure = detect_data_structure(data_dir)
    print(f"Detected data structure: {data_structure}")

    # Handle different data structures
    if data_structure == "imagenet_traditional":
        # Traditional ImageNet with train/val directories
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        
        print(f"Loading ImageNet from traditional structure: {data_dir}")
        try:
            train_dataset = datasets.ImageFolder(
                root=train_dir,
                transform=AlbumentationsAdapter(train_transforms)
            )
            test_dataset = datasets.ImageFolder(
                root=val_dir, 
                transform=AlbumentationsAdapter(test_transforms)
            )
            
            print(f"✓ Traditional ImageNet datasets loaded successfully")
            print(f"✓ Train classes: {len(train_dataset.classes)}")
            print(f"✓ Train samples: {len(train_dataset)}")
            print(f"✓ Val samples: {len(test_dataset)}")
            
            # Apply max_samples if specified
            if max_samples:
                print(f"✓ Limiting to {max_samples} samples per split")
                train_indices = list(range(min(max_samples, len(train_dataset))))
                test_indices = list(range(min(max_samples, len(test_dataset))))
                train_dataset = Subset(train_dataset, train_indices)
                test_dataset = Subset(test_dataset, test_indices)
            
            return train_dataset, test_dataset
            
        except Exception as e:
            print(f"Failed to load traditional ImageNet data: {e}")
    
    elif data_structure == "imagenet_subset":
        # ImageNet subset with numeric class directories (your case)
        print(f"Loading ImageNet subset from: {data_dir}")
        try:
            # Load the full dataset
            full_dataset = ImageNetSubsetDataset(
                root_dir=data_dir,
                transform=None  # We'll apply transforms after splitting
            )
            
            # Create train/val split
            train_dataset, val_dataset = create_train_val_split(
                full_dataset, val_ratio=val_ratio, seed=42
            )
            
            # Apply transforms to the split datasets
            # We need to wrap them to apply transforms
            class TransformWrapper(Dataset):
                def __init__(self, dataset, transform):
                    self.dataset = dataset
                    self.transform = transform
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    image, label = self.dataset[idx]
                    if self.transform:
                        image = self.transform(image)
                    return image, label
                
                @property
                def classes(self):
                    if hasattr(self.dataset, 'dataset'):
                        return self.dataset.dataset.classes
                    return getattr(self.dataset, 'classes', [])
            
            train_dataset = TransformWrapper(train_dataset, AlbumentationsAdapter(train_transforms))
            test_dataset = TransformWrapper(val_dataset, AlbumentationsAdapter(test_transforms))
            
            print(f"✓ ImageNet subset loaded successfully")
            print(f"✓ Classes: {len(full_dataset.classes)}")
            print(f"✓ Train samples: {len(train_dataset)}")
            print(f"✓ Val samples: {len(test_dataset)}")
            
            # Apply max_samples if specified
            if max_samples:
                print(f"✓ Limiting to {max_samples} samples per split")
                train_indices = list(range(min(max_samples, len(train_dataset))))
                test_indices = list(range(min(max_samples, len(test_dataset))))
                train_dataset = Subset(train_dataset, train_indices)
                test_dataset = Subset(test_dataset, test_indices)
            
            return train_dataset, test_dataset
            
        except Exception as e:
            print(f"Failed to load ImageNet subset data: {e}")
    
    elif data_structure == "generic_classes":
        # Generic class directories (fallback to ImageFolder)
        print(f"Loading generic class dataset from: {data_dir}")
        try:
            full_dataset = datasets.ImageFolder(
                root=data_dir,
                transform=None
            )
            
            # Create train/val split
            train_dataset, test_dataset = create_train_val_split(
                full_dataset, val_ratio=val_ratio, seed=42
            )
            
            # Apply transforms after splitting (same wrapper as above)
            class TransformWrapper(Dataset):
                def __init__(self, dataset, transform):
                    self.dataset = dataset
                    self.transform = transform
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    image, label = self.dataset[idx]
                    if self.transform:
                        image = self.transform(image)
                    return image, label
                
                @property
                def classes(self):
                    if hasattr(self.dataset, 'dataset'):
                        return self.dataset.dataset.classes
                    return getattr(self.dataset, 'classes', [])
            
            train_dataset = TransformWrapper(train_dataset, AlbumentationsAdapter(train_transforms))
            test_dataset = TransformWrapper(test_dataset, AlbumentationsAdapter(test_transforms))
            
            print(f"✓ Generic class dataset loaded successfully")
            print(f"✓ Classes: {len(full_dataset.classes)}")
            print(f"✓ Train samples: {len(train_dataset)}")
            print(f"✓ Val samples: {len(test_dataset)}")
            
            # Apply max_samples if specified
            if max_samples:
                print(f"✓ Limiting to {max_samples} samples per split")
                train_indices = list(range(min(max_samples, len(train_dataset))))
                test_indices = list(range(min(max_samples, len(test_dataset))))
                train_dataset = Subset(train_dataset, train_indices)
                test_dataset = Subset(test_dataset, test_indices)
            
            return train_dataset, test_dataset
            
        except Exception as e:
            print(f"Failed to load generic class data: {e}")
    
    # Fallback to Hugging Face datasets if available
    if DATASETS_AVAILABLE:
        try:
            print("Loading ImageNet-1K from Hugging Face datasets...")
            
            # Load ImageNet-1K with streaming
            if streaming:
                print("Using streaming mode - no full dataset download required")
                train_hf_dataset = load_dataset("imagenet-1k", split="train", streaming=True)
                val_hf_dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
            else:
                print("Loading full dataset (this may take a while and require significant disk space)")
                train_hf_dataset = load_dataset("imagenet-1k", split="train", streaming=False)
                val_hf_dataset = load_dataset("imagenet-1k", split="validation", streaming=False)
            
            # Create custom dataset wrappers
            train_dataset = HuggingFaceImageNetDataset(
                train_hf_dataset, 
                transform=train_transforms, 
                max_samples=max_samples
            )
            test_dataset = HuggingFaceImageNetDataset(
                val_hf_dataset, 
                transform=test_transforms, 
                max_samples=max_samples
            )
            
            print(f"✓ ImageNet-1K datasets loaded successfully")
            if max_samples:
                print(f"✓ Limited to {max_samples} samples per split")
            
        except Exception as e:
            print(f"Failed to load ImageNet-1K from Hugging Face: {e}")
            print("Falling back to dummy dataset...")
            # Fallback: create a dummy dataset for testing
            from torch.utils.data import TensorDataset
            import torch
            
            # Create dummy data (1000 classes, 224x224 images)
            dummy_train_data = torch.randn(1000, 3, 224, 224)
            dummy_train_labels = torch.randint(0, 1000, (1000,))
            dummy_test_data = torch.randn(200, 3, 224, 224)
            dummy_test_labels = torch.randint(0, 1000, (200,))
            
            train_dataset = TensorDataset(dummy_train_data, dummy_train_labels)
            test_dataset = TensorDataset(dummy_test_data, dummy_test_labels)
    else:
        print("datasets library not available. Creating dummy dataset...")
        # Fallback: create a dummy dataset for testing
        from torch.utils.data import TensorDataset
        import torch
        
        # Create dummy data (1000 classes, 224x224 images)
        dummy_train_data = torch.randn(1000, 3, 224, 224)
        dummy_train_labels = torch.randint(0, 1000, (1000,))
        dummy_test_data = torch.randn(200, 3, 224, 224)
        dummy_test_labels = torch.randint(0, 1000, (200,))
        
        train_dataset = TensorDataset(dummy_train_data, dummy_train_labels)
        test_dataset = TensorDataset(dummy_test_data, dummy_test_labels)

    return train_dataset, test_dataset


def get_data_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    streaming: bool = True,
    max_samples: int = None,
    data_dir: str = "./data",
    val_ratio: float = 0.2,
):
    """Return ImageNet-1K train/test dataloaders with appropriate transforms."""
    train_dataset, test_dataset = get_datasets(
        streaming=streaming,
        max_samples=max_samples,
        data_dir=data_dir,
        val_ratio=val_ratio
    )

    # For streaming datasets, we need to handle them differently
    if streaming:
        # Create custom dataloader for streaming
        train_loader = StreamingDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        test_loader = StreamingDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        # Standard dataloader for non-streaming datasets
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, test_loader


class StreamingDataLoader:
    """Custom DataLoader for streaming datasets."""
    
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def __iter__(self):
        """Create batches from streaming dataset."""
        batch = []
        labels = []
        
        for data, label in self.dataset:
            batch.append(data)
            labels.append(label)
            
            if len(batch) == self.batch_size:
                # Convert to tensors
                batch_tensor = torch.stack(batch)
                labels_tensor = torch.tensor(labels, dtype=torch.long)
                
                if self.pin_memory and torch.cuda.is_available():
                    batch_tensor = batch_tensor.pin_memory()
                    labels_tensor = labels_tensor.pin_memory()
                
                yield batch_tensor, labels_tensor
                
                # Reset batch
                batch = []
                labels = []
        
        # Yield remaining samples if any
        if batch:
            batch_tensor = torch.stack(batch)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            if self.pin_memory and torch.cuda.is_available():
                batch_tensor = batch_tensor.pin_memory()
                labels_tensor = labels_tensor.pin_memory()
            
            yield batch_tensor, labels_tensor
    
    def __len__(self):
        """Return approximate length based on max_samples."""
        if hasattr(self.dataset, 'max_samples') and self.dataset.max_samples:
            return (self.dataset.max_samples + self.batch_size - 1) // self.batch_size
        return 1000  # Default length for streaming (approximate)
