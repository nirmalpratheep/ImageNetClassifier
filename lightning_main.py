#!/usr/bin/env python3
"""
Minimal PyTorch Lightning trainer with LR finder for ImageNet format data.
Usage: python lightning_main.py --data_dir ./data --batch_size 256
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from model_resnet50 import ResNet50
from PIL import Image
import numpy as np


class TinyImageNetDataset(Dataset):
    """Custom dataset for TinyImageNet with nested images/ folder structure."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        # Get all class folders
        class_folders = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        
        for idx, class_name in enumerate(class_folders):
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx
            
            # Get images from the images subfolder
            images_dir = os.path.join(root_dir, class_name, 'images')
            if os.path.exists(images_dir):
                for img_name in os.listdir(images_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(images_dir, img_name)
                        self.samples.append((img_path, idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class ImageNetLightningModule(pl.LightningModule):
    """Enhanced Lightning module for ImageNet classification with comprehensive logging."""
    
    def __init__(self, num_classes=1000, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = ResNet50(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        # Log basic metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        # Log learning rate-> taken care of by lr_monitor callback
        # current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # self.log('learning_rate', current_lr, prog_bar=True)
        
        # Log gradient norms
        if batch_idx % 50 == 0:  # Log every 50 steps to avoid overhead
            self._log_gradient_norms()
            self._log_model_parameters()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def _log_gradient_norms(self):
        """Log gradient norms for all parameters."""
        total_norm = 0
        param_norms = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                # Log individual parameter gradient norms (for key layers)
                if 'conv' in name or 'fc' in name or 'bn' in name:
                    param_norms[f'grad_norm/{name}'] = param_norm
        
        total_norm = total_norm ** (1. / 2)
        self.log('grad_norm/total', total_norm)
        
        # Log individual parameter norms
        for name, norm in param_norms.items():
            self.log(name, norm)
    
    def _log_model_parameters(self):
        """Log model parameter statistics."""
        param_stats = {}
        
        for name, param in self.named_parameters():
            if 'conv' in name or 'fc' in name or 'bn' in name:  # Log key layers
                param_stats[f'param_mean/{name}'] = param.data.mean().item()
                param_stats[f'param_std/{name}'] = param.data.std().item()
                param_stats[f'param_norm/{name}'] = param.data.norm(2).item()
        
        # Log parameter statistics
        for name, value in param_stats.items():
            self.log(name, value)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch"
        }
    }

def get_imagenet_transforms():
    """Get ImageNet transforms for training and validation."""
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def get_tinyimagenet_transforms():
    """Get TinyImageNet transforms for training and validation."""
    train_transforms = transforms.Compose([
        transforms.Resize(72),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def get_imagenet_dataloaders(data_dir, batch_size=256, num_workers=4):
    """Load ImageNet format data (train/val folders with class subfolders)."""
    train_transforms, val_transforms = get_imagenet_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transforms
    )
    
    # Create dataloaders
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
    
    return train_loader, val_loader, len(train_dataset.classes)


def get_tinyimagenet_dataloaders(data_dir, batch_size=256, num_workers=4):
    """Load TinyImageNet format data with nested images/ folder structure."""
    train_transforms, val_transforms = get_tinyimagenet_transforms()
    
    # Load datasets using custom dataset class
    train_dataset = TinyImageNetDataset(
        root_dir=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    
    val_dataset = TinyImageNetDataset(
        root_dir=os.path.join(data_dir, 'val'),
        transform=val_transforms
    )
    
    # Create dataloaders
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
    
    return train_loader, val_loader, len(train_dataset.classes)


def main():
    parser = argparse.ArgumentParser(description="Minimal Lightning trainer with LR finder")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "tinyimagenet"], 
                       help="Dataset type: imagenet or tinyimagenet")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--max_epochs", type=int, default=10, help="Max epochs")
    parser.add_argument("--lr_finder", action="store_true", help="Run LR finder")
    parser.add_argument("--plot_lr", action="store_true", help="Plot LR finder results")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clip value")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", action="store_true", 
                   help="Resume from latest checkpoint in results/checkpoints/")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                   help="Specific checkpoint path to resume from (overrides --resume)")
    parser.add_argument("--learning_rate", type=float, default=0.1, 
                   help="Learning rate (ignored if --lr_finder is used)")
    parser.add_argument("--results_dir", type=str, default="./results", 
                   help="Directory to store all results (checkpoints, logs, plots)")
    args = parser.parse_args()
    
    print("="*70)
    print("MINIMAL LIGHTNING TRAINER WITH LR FINDER")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Dataset type: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"LR finder: {args.lr_finder}")
    print(f"Results directory: {args.results_dir}")
    print("="*70)
    
    # Create results directory structure
    results_dir = args.results_dir
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    logs_dir = os.path.join(results_dir, "logs")
    plots_dir = os.path.join(results_dir, "plots")
    
    # Create all directories
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"📁 Created results directory structure:")
    print(f"   Checkpoints: {checkpoints_dir}")
    print(f"   Logs: {logs_dir}")
    print(f"   Plots: {plots_dir}")
    print("="*70)
    
    available_gpus = torch.cuda.device_count()
    if available_gpus > 1:
        training_accelerator = "gpu"
        training_devices = available_gpus
        training_strategy = "ddp"
        precision = "bf16-mixed"
        effective_batch_size = args.batch_size * available_gpus
        print(f"🚀 Auto-detected {available_gpus} GPUs - using multi-GPU training")
    elif available_gpus == 1:
        training_accelerator = "gpu"
        training_devices = 1
        training_strategy = "auto"
        precision = "bf16-mixed"
        effective_batch_size = args.batch_size
        print(f"Auto-detected 1 GPU - using single GPU training")
    else:
        training_accelerator = "cpu"
        training_devices = "auto"
        training_strategy = "auto"
        precision = "32"
        effective_batch_size = args.batch_size
        print("No GPUs detected - using CPU training")
    print(f"Effective batch size: {effective_batch_size}")
    # Load data based on dataset type
    if args.dataset == "tinyimagenet":
        print("Loading TinyImageNet data...")
        train_loader, val_loader, num_classes = get_tinyimagenet_dataloaders(
            args.data_dir, args.batch_size, args.num_workers
        )
    else:  # imagenet
        print("Loading ImageNet data...")
        train_loader, val_loader, num_classes = get_imagenet_dataloaders(
            args.data_dir, args.batch_size, args.num_workers
        )
    
    print(f"✓ Found {num_classes} classes")
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Val samples: {len(val_loader.dataset)}")
    
    # Create model
    model = ImageNetLightningModule(num_classes=num_classes,learning_rate=args.learning_rate)
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="model-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=3,  
        every_n_epochs=1,  
        monitor="val_loss",
        mode="min",
        save_on_train_epoch_end=False,
        save_last=True
    )
    
    # Create loggers
    tensorboard_logger = TensorBoardLogger(
        save_dir=logs_dir,
        name="tensorboard_logs",
        version=None,  
        log_graph=True, 
        default_hp_metric=False
    )
    
    csv_logger = CSVLogger(
        save_dir=logs_dir,
        name="csv_logs",
        version=None  
    )
    
    # Create learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    print(f"📊 Logging Configuration:")
    print(f"   TensorBoard logs: {tensorboard_logger.log_dir}")
    print(f"   CSV logs: {csv_logger.log_dir}")
    print(f"   Learning rate monitoring: enabled (per step)")
    print(f"   Gradient norms tracking: enabled (every 50 steps)")
    print(f"   Model parameters tracking: enabled (every 50 steps)")
    print("="*70)

    # if available_gpus >0:
    #     effective_batch_size = args.batch_size * available_gpus
    #     print(f"Effective batch size with {available_gpus} GPUs: {effective_batch_size}")
    # else:
    #     effective_batch_size = args.batch_size
    #     print
    #     args.strategy = "auto"


    if args.lr_finder:
        print("RUNNING LR FINDER")
        print("="*50)
        
        # For LR finder, always use single GPU with auto strategy
        lr_finder_accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        lr_finder_devices = 1 if torch.cuda.is_available() else "auto"
        lr_finder_strategy = "auto"
        
        print(f"🔍 LR Finder Configuration:")
        print(f"   Using single GPU for LR finder (accelerator={lr_finder_accelerator}, devices={lr_finder_devices}, strategy={lr_finder_strategy})")
        
        # Create LR finder trainer with single GPU
        lr_finder_trainer = pl.Trainer(
            max_epochs=1,  # Just for LR finder
            accelerator=lr_finder_accelerator,
            devices=lr_finder_devices,
            precision=precision,
            log_every_n_steps=50,
            val_check_interval=1.0,
            gradient_clip_val=args.gradient_clip_val,
            strategy=lr_finder_strategy,
            enable_checkpointing=False,  # No checkpoints during LR finder
            enable_progress_bar=True,
            enable_model_summary=False
        )
        total_samples = len(train_loader.dataset)  # Use actual train dataset length
        lr_finder_batch_size = args.batch_size  # Use original batch size for LR finder
        steps_per_epoch = total_samples // lr_finder_batch_size
        if total_samples % lr_finder_batch_size != 0:
            steps_per_epoch += 1  # Round up if there's a remainder
        
        print(f"   Total samples: {total_samples:,}")
        print(f"   LR finder batch size: {lr_finder_batch_size}")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Will run LR finder for {steps_per_epoch} steps (1 epoch)")
        
        # Run LR finder
        tuner = Tuner(lr_finder_trainer)
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=train_loader,
            min_lr=1e-6,
            max_lr=1.0,
            num_training=steps_per_epoch,  # Number of steps to run
        )
        
        # Get suggested LR
        suggested_lr = lr_finder.suggestion()
        print(f"Suggested learning rate: {suggested_lr:.2e}")
        
        # Update model with suggested LR
        model.learning_rate = suggested_lr
        print(f"✓ Updated model learning rate to: {suggested_lr:.2e}")
        
        # Plot results if requested
        if args.plot_lr:
            lr_plot_path = os.path.join(plots_dir, "lr_finder_plot.png")
            fig = lr_finder.plot(suggest=True)
            fig.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 LR finder plot saved to: {lr_plot_path}")
        
        print("="*50)
        
    print(f"\n🚀 Creating training trainer with {training_devices} device(s)...")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=training_accelerator,
        devices=training_devices,
        precision=precision,
        log_every_n_steps=50,
        val_check_interval=1.0,
        gradient_clip_val=args.gradient_clip_val,
        strategy=training_strategy,
        default_root_dir=logs_dir,
        accumulate_grad_batches=1,
        logger=[tensorboard_logger, csv_logger],
        callbacks=[
            checkpoint_callback, 
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
            lr_monitor
        ]
    )
    
    # Handle checkpoint resuming
    checkpoint_path = None
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        print(f"\nResuming training from specified checkpoint: {checkpoint_path}")
    elif args.resume:
        # Find the latest checkpoint in checkpoints directory
        import glob
        checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.ckpt"))
        if checkpoint_files:
            # Sort by modification time, get the latest
            checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
            print(f"\nResuming training from latest checkpoint: {checkpoint_path}")
        else:
            print(f"\nNo checkpoints found in {checkpoints_dir}, starting fresh training...")
    
    if checkpoint_path:
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        print(f"\n🚀 Starting training...")
        trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")
    print("="*70)
    print(f"📁 All results saved to: {results_dir}")
    print(f"   Checkpoints: {checkpoints_dir}")
    print(f"   Logs: {logs_dir}")
    print(f"   Plots: {plots_dir}")
    print(f"   TensorBoard logs: {tensorboard_logger.log_dir}")
    print(f"   CSV logs: {csv_logger.log_dir}")
    print("="*70)
    print("📊 To view TensorBoard logs, run:")
    print(f"   tensorboard --logdir {tensorboard_logger.log_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
