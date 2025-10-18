#!/usr/bin/env python3
"""
Debug script to check LR finder plot generation
"""

import argparse
import sys
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Debug LR Finder Plot Generation")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples (small for quick test)")
    parser.add_argument("--lr_iter", type=int, default=20, help="LR iterations (small for quick test)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_path = os.path.join(args.output_dir, "debug_lr_plot.png")
    
    print("="*70)
    print("DEBUG LR FINDER PLOT GENERATION")
    print("="*70)
    print(f"Output dir: {args.output_dir}")
    print(f"Plot path: {plot_path}")
    print(f"Plot path exists before: {os.path.exists(plot_path)}")
    print(f"Output dir exists: {os.path.exists(args.output_dir)}")
    print(f"Output dir writable: {os.access(args.output_dir, os.W_OK)}")
    print("="*70)
    
    # Build minimal command for debugging
    cmd = [
        "uv", "run", "python", "main.py",
        "--batch_size", "32",  # Small batch
        "--find_lr",
        "--lr_start", "1e-4",
        "--lr_end", "1.0",
        "--lr_iter", str(args.lr_iter),  # Few iterations
        "--lr_plot", plot_path,  # Explicit path
        "--data_dir", args.data_dir,
        "--max_samples", str(args.max_samples),  # Very small dataset
        "--val_ratio", "0.3",
        "--epochs", "1",
        "--no_cuda"  # Force CPU for consistent behavior
    ]
    
    print("Running debug command:")
    print(" ".join(cmd))
    print()
    
    # Execute and capture everything
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print()
    
    print("STDERR:")  
    print(result.stderr)
    print()
    
    print("="*70)
    print("POST-EXECUTION CHECK")
    print("="*70)
    print(f"Return code: {result.returncode}")
    print(f"Plot path exists after: {os.path.exists(plot_path)}")
    
    if os.path.exists(plot_path):
        file_size = os.path.getsize(plot_path)
        print(f"Plot file size: {file_size} bytes")
        print("✅ SUCCESS: Plot file was created!")
    else:
        print("❌ FAILED: Plot file was not created")
        
        # Check for any files in output directory
        if os.path.exists(args.output_dir):
            files = os.listdir(args.output_dir)
            print(f"Files in output directory: {files}")
        
        # Check for matplotlib backend issues
        try:
            import matplotlib
            print(f"Matplotlib backend: {matplotlib.get_backend()}")
        except ImportError:
            print("Matplotlib not available")
            
        # Look for any .png files in current directory
        png_files = [f for f in os.listdir('.') if f.endswith('.png')]
        print(f"PNG files in current directory: {png_files}")
    
    # Check for specific error patterns in output
    if "LR finder plot saved to:" in result.stdout:
        print("✅ Code reached the plot saving section")
    else:
        print("❌ Code may not have reached plot saving")
        
    if "Warning: Could not create matplotlib figure" in result.stdout:
        print("❌ Matplotlib figure creation failed")
        
    if "Warning: Could not create plot:" in result.stdout:
        print("❌ Plot creation encountered an error")

if __name__ == "__main__":
    main()
