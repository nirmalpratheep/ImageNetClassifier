#!/usr/bin/env python3
"""
Script to run LR finder on ImageNet1K tiny dataset.
This script demonstrates how to use the LR finder functionality.
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run LR Finder on ImageNet1K")
    parser.add_argument("--dataset", type=str, default="imagenet1k", choices=["cifar100", "imagenet1k"], 
                       help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for LR finder")
    parser.add_argument("--lr_start", type=float, default=1e-7, help="Starting learning rate")
    parser.add_argument("--lr_end", type=float, default=10, help="Ending learning rate")
    parser.add_argument("--lr_iter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--lr_advanced", action="store_true", help="Use advanced LR finder")
    parser.add_argument("--lr_step_mode", type=str, default="exp", choices=["exp", "linear"], help="LR step mode")
    parser.add_argument("--lr_smooth_f", type=float, default=0.05, help="Smoothing factor")
    parser.add_argument("--lr_diverge_th", type=float, default=5, help="Divergence threshold")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--streaming", action="store_true", default=True, help="Use streaming for large datasets")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum samples for LR finder (default: 1000)")
    parser.add_argument("--use_pretrained", action="store_true", help="Use pretrained ResNet-50 weights")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build command
    cmd = [
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
        "--epochs", "1",  # Just run LR finder, no training
        "--no_plots"  # Disable other plots for LR finder run
    ]
    
    if args.streaming:
        cmd.append("--streaming")
    
    if args.use_pretrained:
        cmd.append("--use_pretrained")
    
    if args.lr_advanced:
        cmd.append("--lr_advanced")
    
    if args.no_cuda:
        cmd.append("--no_cuda")
    
    print("Running LR Finder with command:")
    print(" ".join(cmd))
    print()
    
    # Execute command
    import subprocess
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"\nLR Finder completed successfully!")
        print(f"Plot saved to: {os.path.join(args.output_dir, f'lr_finder_{args.dataset}.png')}")
    else:
        print(f"\nLR Finder failed with return code: {result.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
