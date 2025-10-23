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
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from model_resnet50 import ResNet50
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


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
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clip value")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                   help="Path to checkpoint to resume from")
    parser.add_argument("--learning_rate", type=float, default=0.1, 
                   help="Learning rate (ignored if --lr_finder is used)")
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
    
    available_gpus = torch.cuda.device_count()
    if available_gpus > 1:
        accelerator = "gpu"
        devices = available_gpus
        strategy = "ddp"
        precision = "16-mixed"
        effective_batch_size = args.batch_size * available_gpus
        print(f"🚀 Auto-detected {available_gpus} GPUs - using multi-GPU training")
    elif available_gpus == 1:
        accelerator = "gpu"
        devices = 1
        strategy = "auto"
        precision = "16-mixed"
        effective_batch_size = args.batch_size
        print(f"Auto-detected 1 GPU - using single GPU training")
    else:
        accelerator = "cpu"
        devices = "auto"
        strategy = "auto"
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
    dirpath=args.checkpoint_dir,
    filename="model-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,  
    every_n_epochs=1,  
    monitor="val_loss",
    mode="min",
    save_on_train_epoch_end=False,
    save_last=True
)

    # if available_gpus >0:
    #     effective_batch_size = args.batch_size * available_gpus
    #     print(f"Effective batch size with {available_gpus} GPUs: {effective_batch_size}")
    # else:
    #     effective_batch_size = args.batch_size
    #     print
    #     args.strategy = "auto"

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,  # Mixed precision
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
        gradient_clip_val=args.gradient_clip_val,
        strategy=strategy,
        callbacks=[checkpoint_callback,EarlyStopping(monitor="val_loss", mode="min", patience=10)]
    )
    
    if args.lr_finder:
        print("RUNNING LR FINDER")
        print("="*50)
        
        # Calculate steps for 1 epoch based on effective batch size
        total_samples = len(train_loader.dataset)
        steps_per_epoch = total_samples // effective_batch_size
        if total_samples % effective_batch_size != 0:
            steps_per_epoch += 1  # Round up if there's a remainder
        
        print(f" LR Finder Configuration:")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Effective batch size: {effective_batch_size}")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Will run LR finder for {steps_per_epoch} steps (1 epoch)")
        
        # Run LR finder
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
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
            fig = lr_finder.plot(suggest=True)
            fig.savefig("lr_finder_plot.png", dpi=300, bbox_inches='tight')
            print("📊 LR finder plot saved to: lr_finder_plot.png")
        
        print("="*50)
    
    if args.resume_from_checkpoint:
        print(f"\n🚀 Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)
    else:
        print(f"\n🚀 Starting training...")
        trainer.fit(model, train_loader, val_loader)
    # Train the model
   
    
    print("Training completed!")
    print("="*70)


if __name__ == "__main__":
    main()
