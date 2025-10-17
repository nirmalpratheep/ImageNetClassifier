#!/usr/bin/env python3
"""
Safe LR Finder Script - Runs LR finder with better default parameters and safety measures.
This version is designed to work well with partial datasets and avoid extremely small learning rates.
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run Safe LR Finder on ImageNet with better defaults")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for LR finder")
    parser.add_argument("--lr_start", type=float, default=1e-5, help="Starting learning rate (default: 1e-5)")
    parser.add_argument("--lr_end", type=float, default=1.0, help="Ending learning rate (default: 1.0)")
    parser.add_argument("--lr_iter", type=int, default=200, help="Number of iterations (default: 200)")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples for testing (default: 5000)")
    parser.add_argument("--lr_advanced", action="store_true", help="Use advanced LR finder")
    parser.add_argument("--lr_step_mode", type=str, default="exp", choices=["exp", "linear"], help="LR step mode")
    parser.add_argument("--lr_smooth_f", type=float, default=0.05, help="Smoothing factor")
    parser.add_argument("--lr_diverge_th", type=float, default=5, help="Divergence threshold")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    # Wandb arguments  
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="imagenet-lr-finder-safe", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="Wandb tags")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("SAFE LR FINDER - WITH IMPROVED DEFAULTS")
    print("="*70)
    print(f"Dataset: ImageNet (from {args.data_dir})")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples: {args.max_samples} (for faster testing)")
    print(f"LR range: {args.lr_start:.2e} to {args.lr_end:.2e}")
    print(f"LR iterations: {args.lr_iter}")
    print(f"Advanced mode: {args.lr_advanced}")
    print("="*70)
    
    # Build command
    cmd = [
        "uv", "run", "python", "main.py",
        "--batch_size", str(args.batch_size),
        "--find_lr",
        "--lr_start", str(args.lr_start),
        "--lr_end", str(args.lr_end), 
        "--lr_iter", str(args.lr_iter),
        "--lr_plot", os.path.join(args.output_dir, "lr_finder_safe_imagenet.png"),
        "--lr_step_mode", args.lr_step_mode,
        "--lr_smooth_f", str(args.lr_smooth_f),
        "--lr_diverge_th", str(args.lr_diverge_th),
        "--data_dir", args.data_dir,
        "--max_samples", str(args.max_samples),  # Limit samples for safer testing
        "--epochs", "1",  # Run 1 full epoch for LR finder
        "--no_plots"  # Disable other plots for LR finder run
    ]
    
    if args.lr_advanced:
        cmd.append("--lr_advanced")
    
    if args.no_cuda:
        cmd.append("--no_cuda")
    
    # Add wandb arguments if enabled
    if args.use_wandb:
        cmd.extend(["--use_wandb"])
        if args.wandb_project:
            cmd.extend(["--wandb_project", args.wandb_project])
        if args.wandb_run_name:
            cmd.extend(["--wandb_run_name", args.wandb_run_name])
        if args.wandb_tags:
            cmd.extend(["--wandb_tags"] + args.wandb_tags)
    
    print("Running Safe LR Finder with command:")
    print(" ".join(cmd))
    print()
    
    # Execute command
    import subprocess
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✅ Safe LR Finder completed successfully!")
        print(f"📊 Plot saved to: {os.path.join(args.output_dir, 'lr_finder_safe_imagenet.png')}")
        print(f"💡 Tip: Use the suggested LR (or slightly lower) for training")
        print(f"🚀 Next: Run training with the found LR using train_with_lr.py")
    else:
        print(f"\n❌ Safe LR Finder failed with return code: {result.returncode}")
        print(f"💡 Try: Check your data directory structure or use --max_samples 1000 for quicker testing")
        sys.exit(1)

if __name__ == "__main__":
    main()
