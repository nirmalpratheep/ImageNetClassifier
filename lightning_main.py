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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from model_resnet50 import ResNet50


class ImageNetLightningModule(pl.LightningModule):
    """Minimal Lightning module for ImageNet classification with LR finder support."""
    
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
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)


def get_imagenet_transforms():
    """Get ImageNet transforms for training and validation."""
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
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
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
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
    """Load TinyImageNet format data (train/val folders with class subfolders)."""
    train_transforms, val_transforms = get_tinyimagenet_transforms()
    
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
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    
    args = parser.parse_args()
    
    print("="*70)
    print("MINIMAL LIGHTNING TRAINER WITH LR FINDER")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Dataset type: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"LR finder: {args.lr_finder}")
    print("="*70)
    
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
    model = ImageNetLightningModule(num_classes=num_classes)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus if torch.cuda.is_available() else 0,
        precision=16,  # Mixed precision
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
    )
    
    if args.lr_finder:
        print("\n" + "="*50)
        print("RUNNING LR FINDER")
        print("="*50)
        
        # Run LR finder
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            min_lr=1e-6,
            max_lr=1.0,
            num_training=100,  # Number of steps to run
        )
        
        # Get suggested LR
        suggested_lr = lr_finder.suggestion()
        print(f"🎯 Suggested learning rate: {suggested_lr:.2e}")
        
        # Update model with suggested LR
        model.learning_rate = suggested_lr
        print(f"✓ Updated model learning rate to: {suggested_lr:.2e}")
        
        # Plot results if requested
        if args.plot_lr:
            fig = lr_finder.plot(suggest=True)
            fig.savefig("lr_finder_plot.png", dpi=300, bbox_inches='tight')
            print("📊 LR finder plot saved to: lr_finder_plot.png")
        
        print("="*50)
    
    # Train the model
    print(f"\n🚀 Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("✅ Training completed!")
    print("="*70)


if __name__ == "__main__":
    main()
