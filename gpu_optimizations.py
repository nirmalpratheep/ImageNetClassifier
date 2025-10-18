#!/usr/bin/env python3
"""
GPU optimization utilities for A10G training
- PyTorch optimizations
- CUDA settings
- Memory management
"""

import torch
import os

def setup_cuda_optimizations():
    """Setup CUDA optimizations for A10G GPU"""
    
    optimizations = []
    
    # Enable cuDNN benchmarking for consistent input sizes
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        optimizations.append("✅ cuDNN benchmark enabled")
    
    # Enable cuDNN deterministic for reproducibility (disable for max speed)
    # torch.backends.cudnn.deterministic = True  # Uncomment if you need reproducibility
    
    # Enable TensorFloat-32 (TF32) on A10G - significant speedup for mixed precision
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
        optimizations.append("✅ TF32 enabled for matmul")
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
        optimizations.append("✅ TF32 enabled for cuDNN")
    
    # Optimize memory allocation
    if hasattr(torch.cuda, 'memory'):
        # Use memory pool for faster allocations
        try:
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use 95% of VRAM
            optimizations.append("✅ Memory fraction set to 95%")
        except:
            pass
    
    # Set optimal environment variables
    env_optimizations = {
        'CUDA_LAUNCH_BLOCKING': '0',  # Async CUDA operations
        'TORCH_CUDNN_V8_API_ENABLED': '1',  # Use cuDNN v8 API
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',  # Optimize memory fragmentation
    }
    
    for key, value in env_optimizations.items():
        os.environ[key] = value
        optimizations.append(f"✅ {key} = {value}")
    
    return optimizations

def optimize_model(model, use_compile=True):
    """Apply model-level optimizations"""
    
    optimizations = []
    
    # Enable mixed precision optimization at model level
    if hasattr(model, 'half'):
        # Don't convert to half here - let AMP handle it
        pass
    
    # PyTorch 2.0+ compilation (if available)
    if use_compile and hasattr(torch, 'compile'):
        try:
            # Compile model for A10G optimization
            model = torch.compile(
                model,
                mode='max-autotune',  # Maximum optimization
                backend='inductor'    # Default backend
            )
            optimizations.append("✅ torch.compile enabled (max-autotune)")
        except Exception as e:
            optimizations.append(f"⚠️  torch.compile failed: {e}")
    else:
        optimizations.append("ℹ️  torch.compile not available (PyTorch < 2.0)")
    
    return model, optimizations

def get_optimal_batch_size(model, input_shape=(3, 224, 224), device='cuda', start_batch=64):
    """
    Find optimal batch size for A10G GPU using binary search
    WARNING: This will temporarily use GPU memory
    """
    
    if not torch.cuda.is_available():
        return start_batch, ["❌ CUDA not available"]
    
    model = model.to(device)
    model.eval()
    
    # Binary search for maximum batch size
    min_batch = 1
    max_batch = 2048  # A10G theoretical maximum
    optimal_batch = start_batch
    
    print("🔍 Finding optimal batch size...")
    
    while min_batch <= max_batch:
        batch_size = (min_batch + max_batch) // 2
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Test forward pass
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            # If successful, try larger batch
            optimal_batch = batch_size
            min_batch = batch_size + 1
            
            print(f"   ✅ Batch size {batch_size} works")
            
        except torch.cuda.OutOfMemoryError:
            # If OOM, try smaller batch
            max_batch = batch_size - 1
            print(f"   ❌ Batch size {batch_size} OOM")
            
        except Exception as e:
            print(f"   ⚠️  Batch size {batch_size} error: {e}")
            break
    
    # Clean up
    del dummy_input
    torch.cuda.empty_cache()
    
    # Conservative recommendation (90% of maximum)
    recommended_batch = int(optimal_batch * 0.9)
    
    return recommended_batch, [
        f"✅ Maximum batch size: {optimal_batch}",
        f"✅ Recommended batch size: {recommended_batch} (90% of max)"
    ]

def monitor_gpu_usage():
    """Print GPU monitoring command"""
    
    commands = [
        "# Monitor GPU utilization in real-time:",
        "watch -n 1 nvidia-smi",
        "",
        "# Monitor GPU memory usage:",
        "nvidia-smi dmon -s um",
        "",
        "# Monitor GPU power and temperature:",
        "nvidia-smi dmon -s pt",
        "",
        "# One-time detailed GPU info:",
        "nvidia-smi -q"
    ]
    
    print("📊 GPU MONITORING COMMANDS:")
    print("="*50)
    for cmd in commands:
        print(cmd)
    print("="*50)

def print_a10g_specs():
    """Print A10G GPU specifications"""
    
    specs = {
        "GPU": "NVIDIA A10G",
        "Memory": "24 GB GDDR6",
        "Memory Bandwidth": "600 GB/s", 
        "CUDA Cores": "9,216",
        "RT Cores": "36 (2nd Gen)",
        "Tensor Cores": "288 (3rd Gen)",
        "Compute Capability": "8.6",
        "Base Clock": "885 MHz",
        "Boost Clock": "1,695 MHz",
        "TDP": "300W",
        "Mixed Precision": "✅ Supported (BF16, FP16)",
        "TensorFloat-32": "✅ Supported",
        "PCIe": "4.0 x16"
    }
    
    print("🔧 A10G GPU SPECIFICATIONS:")
    print("="*50)
    for key, value in specs.items():
        print(f"{key:<20}: {value}")
    print("="*50)

if __name__ == "__main__":
    print("🚀 A10G GPU OPTIMIZATION TOOLKIT")
    print("="*50)
    
    # Show A10G specs
    print_a10g_specs()
    print()
    
    # Setup optimizations
    print("🛠️ SETTING UP CUDA OPTIMIZATIONS:")
    optimizations = setup_cuda_optimizations()
    for opt in optimizations:
        print(f"  {opt}")
    print()
    
    # Show monitoring commands
    monitor_gpu_usage()
    print()
    
    print("💡 USAGE:")
    print("1. Import: from gpu_optimizations import setup_cuda_optimizations, optimize_model")
    print("2. Call setup_cuda_optimizations() before training")
    print("3. Call optimize_model(model) to optimize your model")
    print("4. Use configs_a10g_optimized.py for pre-tuned configurations")
