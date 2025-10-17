#!/usr/bin/env python3
"""
Clean LR Finder Script - Runs 1 full epoch to find optimal learning rate and plots the result.
Usage: python find_lr.py --data_dir ./data --batch_size 256
"""

import argparse
import sys
import os
import subprocess
import re
import json

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

def main():
    parser = argparse.ArgumentParser(description="Find optimal learning rate using 1 full epoch")
    
    # Dataset arguments
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--use_pretrained", action="store_true", help="Use pretrained ResNet-50 weights")
    
    # LR Finder specific arguments
    parser.add_argument("--lr_start", type=float, default=1e-7, help="Starting learning rate")
    parser.add_argument("--lr_end", type=float, default=10, help="Ending learning rate")
    parser.add_argument("--lr_iter", type=int, default=300, help="Number of iterations for LR finder")
    parser.add_argument("--lr_advanced", action="store_true", help="Use advanced LR finder")
    parser.add_argument("--lr_step_mode", type=str, default="exp", choices=["exp", "linear"], help="LR step mode")
    parser.add_argument("--lr_smooth_f", type=float, default=0.05, help="Smoothing factor")
    parser.add_argument("--lr_diverge_th", type=float, default=5, help="Divergence threshold")
    
    # System arguments
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    args = parser.parse_args()
    
    print("="*70)
    print("LEARNING RATE FINDER - 1 FULL EPOCH")
    print("="*70)
    print(f"Dataset: ImageNet-1K")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR range: {args.lr_start:.2e} to {args.lr_end:.2e}")
    print(f"LR iterations: {args.lr_iter}")
    print(f"Use pretrained: {args.use_pretrained}")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build LR finder command
    cmd = [
        "uv", "run", "python", "main.py",
        "--dataset", "imagenet1k",
        "--batch_size", str(args.batch_size),
        "--find_lr",
        "--lr_start", str(args.lr_start),
        "--lr_end", str(args.lr_end),
        "--lr_iter", str(args.lr_iter),
        "--lr_plot", os.path.join(args.output_dir, "lr_finder_imagenet1k.png"),
        "--lr_step_mode", args.lr_step_mode,
        "--lr_smooth_f", str(args.lr_smooth_f),
        "--lr_diverge_th", str(args.lr_diverge_th),
        "--data_dir", args.data_dir,
        "--epochs", "1",  # Run for exactly 1 full epoch
        "--no_plots"  # Disable other plots
    ]
    
    if args.use_pretrained:
        cmd.append("--use_pretrained")
    
    if args.lr_advanced:
        cmd.append("--lr_advanced")
    
    if args.no_cuda:
        cmd.append("--no_cuda")
    
    print("Running LR Finder...")
    print("Command:", " ".join(cmd))
    print()
    
    # Execute LR finder command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ LR Finder failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)
    
    print("✅ LR Finder completed successfully!")
    print("\nOutput:")
    print(result.stdout)
    
    # Extract suggested learning rate
    suggested_lr = extract_lr_from_output(result.stdout)
    if suggested_lr is None:
        print("⚠️  Warning: Could not extract suggested learning rate from output")
        print("Using default learning rate of 0.001")
        suggested_lr = 0.001
    
    print(f"\n🎯 SUGGESTED LEARNING RATE: {suggested_lr:.2e}")
    
    # Create detailed log file
    log_file_path = os.path.join(args.output_dir, "lr_finder_log.txt")
    with open(log_file_path, 'w') as log_file:
        log_file.write("="*80 + "\n")
        log_file.write("LEARNING RATE FINDER LOG\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Timestamp: {subprocess.check_output(['date'], text=True).strip() if os.name != 'nt' else 'Windows'}\n")
        log_file.write(f"Dataset: ImageNet-1K\n")
        log_file.write(f"Data Directory: {args.data_dir}\n")
        log_file.write(f"Batch Size: {args.batch_size}\n")
        log_file.write(f"Epochs: 1 (Full epoch)\n")
        log_file.write(f"Streaming: No (Offline data)\n")
        log_file.write(f"Pretrained: No\n")
        log_file.write(f"LR Range: {args.lr_start:.2e} to {args.lr_end:.2e}\n")
        log_file.write(f"LR Iterations: {args.lr_iter}\n")
        log_file.write(f"LR Step Mode: {args.lr_step_mode}\n")
        log_file.write(f"Smoothing Factor: {args.lr_smooth_f}\n")
        log_file.write(f"Divergence Threshold: {args.lr_diverge_th}\n")
        log_file.write(f"Advanced LR Finder: {args.lr_advanced}\n")
        log_file.write("="*80 + "\n\n")
        
        log_file.write("COMMAND EXECUTED:\n")
        log_file.write(" ".join(cmd) + "\n\n")
        
        log_file.write("STDOUT OUTPUT:\n")
        log_file.write("-" * 40 + "\n")
        log_file.write(result.stdout + "\n")
        
        if result.stderr:
            log_file.write("\nSTDERR OUTPUT:\n")
            log_file.write("-" * 40 + "\n")
            log_file.write(result.stderr + "\n")
        
        log_file.write("\n" + "="*80 + "\n")
        log_file.write("LR FINDER RESULTS\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Suggested Learning Rate: {suggested_lr:.2e}\n")
        log_file.write(f"Return Code: {result.returncode}\n")
        log_file.write(f"Success: {'Yes' if result.returncode == 0 else 'No'}\n")
        log_file.write(f"Plot File: {os.path.join(args.output_dir, 'lr_finder_imagenet1k.png')}\n")
        
        log_file.write("\n" + "="*80 + "\n")
        log_file.write("ANALYSIS\n")
        log_file.write("="*80 + "\n")
        
        # Add analysis based on suggested LR
        if suggested_lr < 1e-4:
            log_file.write("⚠️  WARNING: Very low learning rate detected!\n")
            log_file.write("   - This might indicate the model needs more training\n")
            log_file.write("   - Consider increasing the learning rate range\n")
            log_file.write("   - Check if the model is properly initialized\n")
        elif suggested_lr > 1.0:
            log_file.write("⚠️  WARNING: Very high learning rate detected!\n")
            log_file.write("   - This might cause training instability\n")
            log_file.write("   - Consider using a lower learning rate\n")
            log_file.write("   - Monitor training loss for divergence\n")
        else:
            log_file.write("✅ Learning rate appears to be in a reasonable range\n")
        
        log_file.write(f"\nRecommended next steps:\n")
        log_file.write(f"1. Use learning rate: {suggested_lr:.2e}\n")
        log_file.write(f"2. Monitor training loss and accuracy\n")
        log_file.write(f"3. Adjust learning rate if needed during training\n")
        log_file.write(f"4. Consider using learning rate scheduling\n")
        
        log_file.write("\n" + "="*80 + "\n")
        log_file.write("END OF LOG\n")
        log_file.write("="*80 + "\n")
    
    # Save the suggested LR to a file for the training script
    lr_info = {
        "suggested_lr": suggested_lr,
        "dataset": "imagenet1k",
        "batch_size": args.batch_size,
        "lr_finder_epochs": 1,
        "plot_file": os.path.join(args.output_dir, "lr_finder_imagenet1k.png"),
        "log_file": log_file_path,
        "timestamp": subprocess.check_output(["date"], text=True).strip() if os.name != 'nt' else "Windows"
    }
    
    lr_info_path = os.path.join(args.output_dir, "suggested_lr.json")
    with open(lr_info_path, 'w') as f:
        json.dump(lr_info, f, indent=2)
    
    print(f"\n📊 LR Finder plot saved to: {lr_info['plot_file']}")
    print(f"📝 Detailed log saved to: {log_file_path}")
    print(f"💾 LR info saved to: {lr_info_path}")
    print(f"\n🚀 Next step: Run training with --lr {suggested_lr:.2e}")
    print("="*70)

if __name__ == "__main__":
    main()
