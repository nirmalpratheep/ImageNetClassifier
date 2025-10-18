#!/usr/bin/env python3
"""
Optimized training configurations for AWS g5.2xlarge (A10G GPU)
- 24GB VRAM
- 1 GPU
- High memory bandwidth
- Supports mixed precision training
"""

import os
import subprocess
import argparse

# A10G GPU Optimized Configurations
A10G_CONFIGS = {
    "lr_finder": {
        "description": "LR Finder - Optimized for A10G",
        "batch_size": 512,  # Large batch for stable LR finding
        "max_samples": 5000,
        "lr_iter": 300,
        "num_workers": 8,
        "flags": ["--amp"],  # Mixed precision for faster computation
        "extra": {
            "lr_start": "1e-6",
            "lr_end": "1.0"
        }
    },
    
    "sample_training": {
        "description": "Sample Training - Fast iteration on subset",
        "batch_size": 768,  # Maximize A10G utilization
        "max_samples": 25000,
        "epochs": 20,
        "num_workers": 12,
        "flags": ["--amp"],
        "scheduler": "onecycle",
        "extra": {
            "max_grad_norm": "1.0"
        }
    },
    
    "full_training_conservative": {
        "description": "Full Training - Conservative (safe batch size)",
        "batch_size": 384,  # Conservative for full dataset
        "epochs": 50,
        "num_workers": 16,
        "flags": ["--amp", "--use_wandb"],
        "scheduler": "cosine",
        "extra": {
            "max_grad_norm": "1.0",
            "snapshot_freq": "5",
            "save_best": True
        }
    },
    
    "full_training_aggressive": {
        "description": "Full Training - Aggressive (max batch size)",
        "batch_size": 512,  # Push A10G to limits
        "epochs": 50, 
        "num_workers": 16,
        "flags": ["--amp", "--use_wandb"],
        "scheduler": "cosine",
        "extra": {
            "max_grad_norm": "1.0",
            "gradient_accumulation": "2",  # If we need even larger effective batch
            "snapshot_freq": "5",
            "save_best": True
        }
    },
    
    "speed_benchmark": {
        "description": "Speed Benchmark - Maximum throughput test",
        "batch_size": 1024,  # Extreme batch size
        "max_samples": 10000,
        "epochs": 3,
        "num_workers": 20,
        "flags": ["--amp", "--no_plots"],
        "scheduler": "onecycle",
        "extra": {
            "max_grad_norm": "1.0"
        }
    }
}

def get_system_info():
    """Get system information for optimization"""
    try:
        # CPU cores
        cpu_count = os.cpu_count()
        
        # GPU info
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        
        # Memory info  
        with open('/proc/meminfo', 'r') as f:
            mem_info = f.readlines()
        
        total_ram_kb = int([line for line in mem_info if 'MemTotal' in line][0].split()[1])
        total_ram_gb = total_ram_kb // (1024 * 1024)
        
        return {
            "cpu_cores": cpu_count,
            "gpu_info": gpu_info.stdout.strip() if gpu_info.returncode == 0 else "Unknown",
            "total_ram_gb": total_ram_gb
        }
    except Exception as e:
        return {"error": str(e)}

def build_command(config_name, config, data_dir="./data", script="main.py"):
    """Build optimized command for the configuration"""
    
    cmd = ["uv", "run", "python", script]
    
    # Core parameters
    cmd.extend(["--batch_size", str(config["batch_size"])])
    cmd.extend(["--epochs", str(config["epochs"])])
    cmd.extend(["--num_workers", str(config["num_workers"])])
    cmd.extend(["--data_dir", data_dir])
    
    # Optional parameters
    if "max_samples" in config:
        cmd.extend(["--max_samples", str(config["max_samples"])])
    
    if "scheduler" in config:
        cmd.extend(["--scheduler", config["scheduler"]])
    
    # Flags
    if "flags" in config:
        cmd.extend(config["flags"])
    
    # Extra parameters
    if "extra" in config:
        for key, value in config["extra"].items():
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key}", str(value)])
    
    return cmd

def run_config(config_name, data_dir="./data", dry_run=False):
    """Run a specific configuration"""
    
    if config_name not in A10G_CONFIGS:
        print(f"❌ Unknown config: {config_name}")
        print(f"Available configs: {list(A10G_CONFIGS.keys())}")
        return False
    
    config = A10G_CONFIGS[config_name]
    
    print("="*70)
    print(f"A10G OPTIMIZED CONFIG: {config_name.upper()}")
    print("="*70)
    print(f"Description: {config['description']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Workers: {config['num_workers']}")
    if 'max_samples' in config:
        print(f"Max samples: {config['max_samples']:,}")
    print(f"Epochs: {config['epochs']}")
    print("="*70)
    
    # Determine which script to use
    if config_name == "lr_finder":
        script = "run_lr_finder_clean.py"
        cmd = [
            "uv", "run", "python", script,
            "--batch_size", str(config["batch_size"]),
            "--max_samples", str(config["max_samples"]),
            "--lr_iter", str(config["extra"]["lr_iter"]),
            "--lr_start", config["extra"]["lr_start"],
            "--lr_end", config["extra"]["lr_end"],
            "--data_dir", data_dir
        ]
        if "--amp" in config.get("flags", []):
            print("Note: AMP is enabled by default in LR finder")
    else:
        script = "main.py"
        cmd = build_command(config_name, config, data_dir, script)
    
    print("Command:")
    print(" ".join(cmd))
    print()
    
    if dry_run:
        print("🔍 DRY RUN - Command not executed")
        return True
    
    # Execute the command
    print("🚀 Starting training...")
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user")
        return False

def show_recommendations():
    """Show A10G optimization recommendations"""
    
    print("="*70)
    print("A10G GPU OPTIMIZATION RECOMMENDATIONS")
    print("="*70)
    
    system_info = get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    print()
    
    print("🚀 PERFORMANCE OPTIMIZATIONS:")
    print("1. Mixed Precision (AMP): Always enabled - A10G has Tensor Cores")
    print("2. Batch Size: 384-768 recommended, up to 1024 for small datasets")
    print("3. Workers: 12-16 for data loading (g5.2xlarge has 8 vCPUs)")
    print("4. Memory: 24GB VRAM allows large batch sizes")
    print("5. Scheduler: OneCycleLR for fast training, Cosine for stability")
    print()
    
    print("🎯 CONFIGURATION GUIDE:")
    print("• lr_finder: Find optimal LR with large batch")
    print("• sample_training: Fast iteration on 25k samples") 
    print("• full_training_conservative: Safe full training (384 batch)")
    print("• full_training_aggressive: Max performance (512+ batch)")
    print("• speed_benchmark: Test maximum throughput")
    print()
    
    print("💡 TIPS:")
    print("• Monitor GPU utilization: watch -n 1 nvidia-smi")
    print("• If OOM, reduce batch_size by 25% and retry")
    print("• Use gradient accumulation if you need larger effective batch")
    print("• Enable wandb for experiment tracking on long runs")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="A10G GPU Optimized Training Configs")
    parser.add_argument("config", nargs="?", help="Configuration to run", 
                       choices=list(A10G_CONFIGS.keys()) + ["recommendations", "list"])
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--dry_run", action="store_true", help="Show command without executing")
    
    args = parser.parse_args()
    
    if args.config == "recommendations" or args.config is None:
        show_recommendations()
    elif args.config == "list":
        print("Available A10G configurations:")
        for name, config in A10G_CONFIGS.items():
            print(f"  {name:<25} - {config['description']}")
    else:
        success = run_config(args.config, args.data_dir, args.dry_run)
        if not success and not args.dry_run:
            exit(1)

if __name__ == "__main__":
    main()
