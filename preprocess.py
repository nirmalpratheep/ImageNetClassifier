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

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


# Dataset statistics (RGB)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# ImageNet statistics (RGB)
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


def get_transforms(dataset_name: str = "cifar100"):
    """Return transforms for different datasets."""
    if dataset_name.lower() == "imagenet1k" or dataset_name.lower() == "imagenet":
        # ImageNet transforms
        fill_value = _coarse_dropout_fill_value_from_mean(IMAGENET_MEAN)
        
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
    
    else:
        # CIFAR-100 transforms (default)
        fill_value = _coarse_dropout_fill_value_from_mean(CIFAR100_MEAN)

        train_transforms = Compose([
            PadIfNeeded(min_height=36, min_width=36, border_mode=0, p=1.0),
            RandomCrop(height=32, width=32, p=1.0),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
            CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(16, 16),
                hole_width_range=(16, 16),
                fill=fill_value,
                p=0.4,
            ),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.4),
            Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
            ToTensorV2(),
        ])

        test_transforms = Compose([
            Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
            ToTensorV2(),
        ])

        return AlbumentationsAdapter(train_transforms), AlbumentationsAdapter(test_transforms)


def get_datasets(data_dir: str = "./data", dataset_name: str = "cifar100", streaming: bool = True, max_samples: int = None):
    """Return train/test datasets with appropriate transforms."""
    train_transforms, test_transforms = get_transforms(dataset_name)

    if dataset_name.lower() == "imagenet1k" or dataset_name.lower() == "imagenet":
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
    else:
        # CIFAR-100 (default)
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transforms
        )
        test_dataset = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=test_transforms
        )

    return train_dataset, test_dataset


def get_data_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    dataset_name: str = "cifar100",
    streaming: bool = True,
    max_samples: int = None,
):
    """Return train/test dataloaders with appropriate transforms."""
    train_dataset, test_dataset = get_datasets(
        data_dir=data_dir, 
        dataset_name=dataset_name, 
        streaming=streaming,
        max_samples=max_samples
    )

    # For streaming datasets, we need to handle them differently
    if streaming and dataset_name.lower() in ["imagenet1k", "imagenet"]:
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
