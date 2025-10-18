#!/usr/bin/env python3
"""
Clean LR Finder Script - Same as safe version but with suppressed warnings for cleaner output.
"""

import argparse
import sys
import os
import subprocess
import re
import json
from datetime import datetime
import warnings

# Suppress specific warnings that are not relevant for our use case
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def extract_lr_from_output(output_text):
    """Extract the suggested learning rate from the LR finder output."""
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

def clean_stderr(stderr_text):
    """Filter out known harmless warnings from stderr."""
    lines = stderr_text.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Skip albumentations warnings
        if "albumentations" in line and "ShiftScaleRotate" in line:
            continue
        # Skip sklearn warnings about class distribution
        if "sklearn" in line and ("regression problem" in line or "multiclass" in line):
            continue
        # Skip sklearn warnings about unique classes
        if "number of unique classes is greater than 50%" in line:
            continue
        # Keep other potentially important errors
        if line.strip():
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def main():
    parser = argparse.ArgumentParser(description="Run Clean LR Finder (suppressed warnings)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for LR finder")
    parser.add_argument("--lr_start", type=float, default=1e-5, help="Starting learning rate (default: 1e-5)")
    parser.add_argument("--lr_end", type=float, default=1.0, help="Ending learning rate (default: 1.0)")
    parser.add_argument("--lr_iter", type=int, default=200, help="Number of iterations (default: 200)")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples for testing (default: 5000)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser.add_argument("--lr_advanced", action="store_true", help="Use advanced LR finder")
    parser.add_argument("--lr_step_mode", type=str, default="exp", choices=["exp", "linear"], help="LR step mode")
    parser.add_argument("--lr_smooth_f", type=float, default=0.05, help="Smoothing factor")
    parser.add_argument("--lr_diverge_th", type=float, default=5, help="Divergence threshold")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--show_warnings", action="store_true", help="Show all warnings (default: suppressed)")
    
    # Wandb arguments  
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="imagenet-lr-finder-clean", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="Wandb tags")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("CLEAN LR FINDER - SUPPRESSED WARNINGS")
    print("="*70)
    print(f"Dataset: ImageNet (from {args.data_dir})")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples: {args.max_samples} (for faster testing)")
    print(f"LR range: {args.lr_start:.2e} to {args.lr_end:.2e}")
    print(f"LR iterations: {args.lr_iter}")
    print(f"Advanced mode: {args.lr_advanced}")
    print(f"Show warnings: {args.show_warnings}")
    print("="*70)
    
    # Build command with warning suppression
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore::UserWarning'
    
    cmd = [
        "uv", "run", "python", "main.py",
        "--batch_size", str(args.batch_size),
        "--find_lr",
        "--lr_start", str(args.lr_start),
        "--lr_end", str(args.lr_end), 
        "--lr_iter", str(args.lr_iter),
        "--lr_plot", os.path.join(args.output_dir, "lr_finder_clean_imagenet.png"),
        "--lr_step_mode", args.lr_step_mode,
        "--lr_smooth_f", str(args.lr_smooth_f),
        "--lr_diverge_th", str(args.lr_diverge_th),
        "--data_dir", args.data_dir,
        "--max_samples", str(args.max_samples),
        "--val_ratio", str(args.val_ratio),
        "--epochs", "1",
        # Allow LR finder plot generation (no --no_plots)
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
    
    print("Running Clean LR Finder...")
    if not args.show_warnings:
        print("📢 Note: Warnings are suppressed for cleaner output. Use --show_warnings to see all warnings.")
    print()
    
    # Execute command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    # Print the stdout (main output)
    print(result.stdout)
    
    # Handle stderr - filter warnings if requested
    if result.stderr:
        if args.show_warnings:
            print("STDERR:", result.stderr)
        else:
            cleaned_stderr = clean_stderr(result.stderr)
            if cleaned_stderr.strip():
                print("IMPORTANT ERRORS/WARNINGS:", cleaned_stderr)
    
    if result.returncode == 0:
        # Extract suggested learning rate from output
        suggested_lr = extract_lr_from_output(result.stdout)
        
        if suggested_lr is None:
            print("⚠️  Warning: Could not extract suggested learning rate from output")
            print("Using default learning rate of 0.001")
            suggested_lr = 0.001
        
        print(f"\n🎯 SUGGESTED LEARNING RATE: {suggested_lr:.2e}")
        
        # Create suggested_lr.json file
        lr_info = {
            "suggested_lr": suggested_lr,
            "dataset": "imagenet_subset",
            "batch_size": args.batch_size,
            "lr_finder_epochs": 1,
            "max_samples": args.max_samples,
            "val_ratio": args.val_ratio,
            "lr_range_start": args.lr_start,
            "lr_range_end": args.lr_end,
            "lr_iterations": args.lr_iter,
            "plot_file": os.path.join(args.output_dir, "lr_finder_clean_imagenet.png"),
            "timestamp": datetime.now().isoformat(),
            "warnings_suppressed": not args.show_warnings
        }
        
        # Save suggested LR to JSON file
        lr_info_path = os.path.join(args.output_dir, "suggested_lr.json")
        with open(lr_info_path, 'w') as f:
            json.dump(lr_info, f, indent=2)
        
        print(f"\n✅ Clean LR Finder completed successfully!")
        print(f"📊 Plot saved to: {os.path.join(args.output_dir, 'lr_finder_clean_imagenet.png')}")
        print(f"💾 LR info saved to: {lr_info_path}")
        print(f"🎯 Suggested LR: {suggested_lr:.2e}")
        print(f"💡 Tip: Use the suggested LR (or slightly lower) for training")
        print(f"🚀 Next: Run training with --lr {suggested_lr:.2e} or use train_with_lr.py --auto_lr")
        
        if not args.show_warnings:
            print(f"\n📢 Note: This run suppressed harmless sklearn/albumentations warnings")
            print(f"   These warnings are about class distribution and deprecated transforms - safe to ignore")
            
    else:
        print(f"\n❌ Clean LR Finder failed with return code: {result.returncode}")
        print(f"💡 Try: Check your data directory structure or use --max_samples 1000 for quicker testing")
        sys.exit(1)

if __name__ == "__main__":
    main()
