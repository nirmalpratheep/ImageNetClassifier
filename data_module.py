"""PyTorch Lightning DataModule for ImageNet classification."""
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from dataset_loader import ImageNetDataset, get_transforms


class ImageNetDataModule(LightningDataModule):
    """Lightning DataModule for ImageNet dataset.
    
    Automatically handles DistributedSampler when used with DDP.
    PyTorch Lightning will wrap the sampler for distributed training.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        image_size: int = 224,
        num_workers: int = 4,
        subset_size: int = None,
        augmentation: bool = True,
        max_samples_per_class: int = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.subset_size = subset_size
        self.augmentation = augmentation
        self.max_samples_per_class = max_samples_per_class
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        
        self.train_dataset = None
        self.val_dataset = None
        self.num_classes = None
    
    def setup(self, stage: str = None):
        """Setup datasets for training and validation."""
        if stage == "fit" or stage is None:
            train_dir = os.path.join(self.data_dir, 'train')
            val_dir = os.path.join(self.data_dir, 'val')
            
            train_transform, val_transform = get_transforms(
                image_size=self.image_size,
                augmentation=self.augmentation
            )
            
            print(f"DEBUG DataModule: max_samples_per_class = {self.max_samples_per_class}")
            self.train_dataset = ImageNetDataset(
                data_dir=train_dir,
                transform=train_transform,
                subset_size=self.subset_size,
                max_samples_per_class=self.max_samples_per_class
            )
            print(f"DEBUG DataModule: After creating train_dataset, max_samples_per_class = {self.max_samples_per_class}")
            
            self.val_dataset = ImageNetDataset(
                data_dir=val_dir,
                transform=val_transform,
                subset_size=self.subset_size,
                max_samples_per_class=self.max_samples_per_class
            )
            
            self.num_classes = self.train_dataset.num_classes
            print(f"DataModule setup complete: {self.num_classes} classes")
            print(f"Training samples: {len(self.train_dataset)}")
            print(f"Validation samples: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        """Create training DataLoader with optimized settings.
        
        Note: When using DDP, Lightning automatically uses DistributedSampler
        when shuffle=True is specified. Each GPU will see unique, non-overlapping batches.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Lightning wraps this with DistributedSampler in DDP mode
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
    
    def val_dataloader(self):
        """Create validation DataLoader with optimized settings.
        
        Note: When using DDP, Lightning automatically uses DistributedSampler
        for validation to ensure each GPU sees different samples.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Lightning wraps this with DistributedSampler in DDP mode
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
