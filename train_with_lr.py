#!/usr/bin/env python3
"""
Clean Training Script - Runs 5 epochs using OneCycleLR scheduler with validation after each epoch.
Usage: python train_with_lr.py --dataset imagenet1k --lr 0.001 --batch_size 32
"""

import argparse
import sys
import os
import subprocess
import json

def load_suggested_lr(lr_info_path):
    """Load suggested learning rate from LR finder output."""
    if os.path.exists(lr_info_path):
        with open(lr_info_path, 'r') as f:
            lr_info = json.load(f)
            return lr_info.get('suggested_lr', None)
    return None

def main():
    parser = argparse.ArgumentParser(description="Train model for 5 epochs using OneCycleLR scheduler")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="imagenet1k", choices=["cifar100", "imagenet1k"], 
                       help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples for training (None = full dataset)")
    parser.add_argument("--use_pretrained", action="store_true", help="Use pretrained ResNet-50 weights")
    
    # Learning rate arguments
    parser.add_argument("--lr", type=float, help="Learning rate (required or will load from suggested_lr.json)")
    parser.add_argument("--auto_lr", action="store_true", help="Automatically load LR from suggested_lr.json")
    
    # OneCycleLR specific arguments
    parser.add_argument("--onecycle_pct_start", type=float, default=0.3, help="OneCycleLR: percent of cycle for warmup")
    parser.add_argument("--onecycle_div_factor", type=float, default=25.0, help="OneCycleLR: initial_lr = max_lr/div_factor")
    parser.add_argument("--onecycle_final_div_factor", type=float, default=10000.0, help="OneCycleLR: min_lr = initial_lr/final_div_factor")
    parser.add_argument("--onecycle_anneal_strategy", type=str, default="cos", choices=["cos", "linear"], help="OneCycleLR: annealing strategy")
    
    # Training arguments
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    
    # System arguments
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    
    args = parser.parse_args()
    
    # Determine learning rate
    learning_rate = args.lr
    if learning_rate is None:
        if args.auto_lr:
            lr_info_path = os.path.join(args.output_dir, "suggested_lr.json")
            learning_rate = load_suggested_lr(lr_info_path)
            if learning_rate is None:
                print("❌ Error: Could not find suggested_lr.json file")
                print("Please run find_lr.py first or specify --lr manually")
                sys.exit(1)
            print(f"📖 Loaded suggested LR from file: {learning_rate:.2e}")
        else:
            print("❌ Error: Must specify --lr or use --auto_lr to load from suggested_lr.json")
            sys.exit(1)
    
    print("="*70)
    print("TRAINING WITH ONECYCLELR - 5 EPOCHS")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {learning_rate:.2e}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'FULL DATASET'}")
    print(f"Use pretrained: {args.use_pretrained}")
    print(f"OneCycleLR warmup: {args.onecycle_pct_start*100:.0f}%")
    print(f"OneCycleLR div_factor: {args.onecycle_div_factor}")
    print(f"OneCycleLR final_div_factor: {args.onecycle_final_div_factor}")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build training command
    cmd = [
        "uv", "run", "python", "main.py",
        "--dataset", args.dataset,
        "--batch_size", str(args.batch_size),
        "--lr", str(learning_rate),  # Use the specified/loaded LR
        "--epochs", "5",  # Run for exactly 5 epochs
        "--scheduler", "onecycle",  # Use OneCycleLR scheduler
        "--onecycle_pct_start", str(args.onecycle_pct_start),
        "--onecycle_div_factor", str(args.onecycle_div_factor),
        "--onecycle_final_div_factor", str(args.onecycle_final_div_factor),
        "--onecycle_anneal_strategy", args.onecycle_anneal_strategy,
        "--weight_decay", str(args.weight_decay),
        "--momentum", str(args.momentum),
        "--data_dir", args.data_dir,
        "--snapshot_dir", os.path.join(args.output_dir, "snapshots"),
        "--plot_dir", os.path.join(args.output_dir, "plots"),
        "--plot_training",  # Enable training plots
        "--plot_evaluation",  # Enable evaluation plots
        "--snapshot_freq", "1",  # Save snapshot every epoch
        "--save_best",  # Save best model
        "--streaming"  # Always use streaming
    ]
    
    # Add max_samples only if specified (None means full dataset)
    if args.max_samples is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])
    
    if args.use_pretrained:
        cmd.append("--use_pretrained")
    
    if args.no_cuda:
        cmd.append("--no_cuda")
    
    if args.amp:
        cmd.append("--amp")
    
    print("Running training...")
    print("Command:", " ".join(cmd))
    print()
    
    # Execute training command
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("❌ Training failed!")
        sys.exit(1)
    
    print("\n✅ Training completed successfully!")
    print("="*70)
    print("TRAINING RESULTS")
    print("="*70)
    print(f"📊 Training plots: {os.path.join(args.output_dir, 'plots')}")
    print(f"💾 Model snapshots: {os.path.join(args.output_dir, 'snapshots')}")
    print(f"🎯 Used learning rate: {learning_rate:.2e}")
    print("="*70)

if __name__ == "__main__":
    main()
