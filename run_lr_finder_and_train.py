#!/usr/bin/env python3
"""
Script to run LR finder for 3 epochs, then use the found LR for full training of 5 epochs.
This script demonstrates the complete workflow of LR finding followed by training.
"""

import argparse
import sys
import os
import subprocess
import json
import re

def extract_lr_from_output(output_text):
    """Extract the suggested learning rate from the LR finder output."""
    # Look for patterns like "Suggested learning rate: 1.23e-04"
    patterns = [
        r"Suggested learning rate: ([\d\.e\+\-]+)",
        r"Suggested LR: ([\d\.e\+\-]+)",
        r"Learning rate: ([\d\.e\+\-]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return None

def run_lr_finder(args):
    """Run LR finder for 3 epochs to find optimal learning rate."""
    print("="*70)
    print("STEP 1: RUNNING LEARNING RATE FINDER (3 epochs)")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build LR finder command
    lr_finder_cmd = [
        "uv", "run", "python", "main.py",
        "--dataset", args.dataset,
        "--batch_size", str(args.batch_size),
        "--find_lr",
        "--lr_start", str(args.lr_start),
        "--lr_end", str(args.lr_end),
        "--lr_iter", str(args.lr_iter),
        "--lr_plot", os.path.join(args.output_dir, f"lr_finder_{args.dataset}.png"),
        "--lr_step_mode", args.lr_step_mode,
        "--lr_smooth_f", str(args.lr_smooth_f),
        "--lr_diverge_th", str(args.lr_diverge_th),
        "--data_dir", args.data_dir,
        "--max_samples", str(args.max_samples),
        "--epochs", "3",  # Run for 3 epochs to find LR
        "--no_plots"  # Disable other plots for LR finder run
    ]
    
    if args.streaming:
        lr_finder_cmd.append("--streaming")
    
    if args.use_pretrained:
        lr_finder_cmd.append("--use_pretrained")
    
    if args.lr_advanced:
        lr_finder_cmd.append("--lr_advanced")
    
    if args.no_cuda:
        lr_finder_cmd.append("--no_cuda")
    
    print("Running LR Finder with command:")
    print(" ".join(lr_finder_cmd))
    print()
    
    # Execute LR finder command and capture output
    result = subprocess.run(lr_finder_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"LR Finder failed with return code: {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None
    
    print("LR Finder completed successfully!")
    print("STDOUT:", result.stdout)
    
    # Extract suggested learning rate
    suggested_lr = extract_lr_from_output(result.stdout)
    if suggested_lr is None:
        print("Warning: Could not extract suggested learning rate from output")
        print("Using default learning rate of 0.001")
        suggested_lr = 0.001
    
    print(f"Extracted suggested learning rate: {suggested_lr:.2e}")
    
    # Save the suggested LR to a file for reference
    lr_info = {
        "suggested_lr": suggested_lr,
        "lr_finder_epochs": 3,
        "output_file": os.path.join(args.output_dir, f"lr_finder_{args.dataset}.png")
    }
    
    lr_info_path = os.path.join(args.output_dir, "lr_finder_info.json")
    with open(lr_info_path, 'w') as f:
        json.dump(lr_info, f, indent=2)
    
    print(f"LR finder info saved to: {lr_info_path}")
    print(f"LR finder plot saved to: {lr_info['output_file']}")
    
    return suggested_lr

def run_full_training(args, suggested_lr):
    """Run full training for 5 epochs using the suggested learning rate."""
    print("\n" + "="*70)
    print("STEP 2: RUNNING FULL TRAINING (5 epochs) WITH VALIDATION")
    print("="*70)
    
    # Build training command
    train_cmd = [
        "uv", "run", "python", "main.py",
        "--dataset", args.dataset,
        "--batch_size", str(args.batch_size),
        "--lr", str(suggested_lr),  # Use the suggested LR from step 1
        "--epochs", "5",  # Run for 5 epochs
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--scheduler", args.scheduler,
        "--weight_decay", str(args.weight_decay),
        "--momentum", str(args.momentum),
        "--max_samples", str(args.max_samples),
        "--snapshot_dir", os.path.join(args.output_dir, "snapshots"),
        "--plot_dir", os.path.join(args.output_dir, "plots"),
        "--plot_training",  # Enable training plots
        "--plot_evaluation",  # Enable evaluation plots
        "--snapshot_freq", "1",  # Save snapshot every epoch
        "--save_best"  # Save best model
    ]
    
    if args.streaming:
        train_cmd.append("--streaming")
    
    if args.use_pretrained:
        train_cmd.append("--use_pretrained")
    
    if args.no_cuda:
        train_cmd.append("--no_cuda")
    
    if args.amp:
        train_cmd.append("--amp")
    
    print("Running full training with command:")
    print(" ".join(train_cmd))
    print()
    
    # Execute training command
    result = subprocess.run(train_cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Training failed with return code: {result.returncode}")
        return False
    
    print("Training completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run LR Finder + Full Training Pipeline")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="imagenet1k", choices=["cifar100", "imagenet1k"], 
                       help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--streaming", action="store_true", default=True, help="Use streaming for large datasets")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum samples for LR finder")
    
    # LR Finder arguments
    parser.add_argument("--lr_start", type=float, default=1e-7, help="Starting learning rate for LR finder")
    parser.add_argument("--lr_end", type=float, default=10, help="Ending learning rate for LR finder")
    parser.add_argument("--lr_iter", type=int, default=300, help="Number of iterations for LR finder")
    parser.add_argument("--lr_advanced", action="store_true", help="Use advanced LR finder")
    parser.add_argument("--lr_step_mode", type=str, default="exp", choices=["exp", "linear"], help="LR step mode")
    parser.add_argument("--lr_smooth_f", type=float, default=0.05, help="Smoothing factor")
    parser.add_argument("--lr_diverge_th", type=float, default=5, help="Divergence threshold")
    
    # Training arguments
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "onecycle"], 
                       help="Learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    
    # Model arguments
    parser.add_argument("--use_pretrained", action="store_true", help="Use pretrained ResNet-50 weights")
    
    # System arguments
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    
    args = parser.parse_args()
    
    print("LR Finder + Training Pipeline")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Streaming: {args.streaming}")
    print(f"Max samples: {args.max_samples}")
    print(f"Use pretrained: {args.use_pretrained}")
    print("="*70)
    
    # Step 1: Run LR finder for 3 epochs
    suggested_lr = run_lr_finder(args)
    if suggested_lr is None:
        print("Failed to find learning rate. Exiting.")
        sys.exit(1)
    
    # Step 2: Run full training for 5 epochs
    success = run_full_training(args, suggested_lr)
    if not success:
        print("Training failed. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Suggested LR found: {suggested_lr:.2e}")
    print(f"LR finder plot: {os.path.join(args.output_dir, f'lr_finder_{args.dataset}.png')}")
    print(f"Training snapshots: {os.path.join(args.output_dir, 'snapshots')}")
    print(f"Training plots: {os.path.join(args.output_dir, 'plots')}")
    print(f"LR info saved: {os.path.join(args.output_dir, 'lr_finder_info.json')}")
    print("="*70)

if __name__ == "__main__":
    main()
