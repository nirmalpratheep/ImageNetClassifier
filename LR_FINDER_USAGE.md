# Learning Rate Finder Usage Guide

This guide explains how to use the Learning Rate Finder functionality with ImageNet1K tiny dataset using the `torch-lr-finder` library.

## Overview

The LR Finder helps you find the optimal learning rate for training by:
1. Starting with a very small learning rate
2. Exponentially increasing it during training
3. Monitoring the loss to find the steepest descent point
4. Suggesting the optimal learning rate

## Installation

**torch-lr-finder and datasets are REQUIRED** for this implementation. Install them with:

```bash
pip install torch-lr-finder datasets
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

**Note**: 
- This implementation uses ONLY torch-lr-finder library. There is no fallback to custom implementation.
- ImageNet-1K is loaded via Hugging Face datasets with streaming support (no full download required).

## Quick Start

### 1. Run LR Finder on ImageNet1K with streaming

```bash
python run_lr_finder.py --dataset imagenet1k --batch_size 32 --max_samples 1000
```

**Note**: The dataset is streamed from Hugging Face - no local download required!

### 2. Run LR Finder with custom parameters

```bash
python run_lr_finder.py \
    --dataset imagenet1k \
    --batch_size 64 \
    --max_samples 2000 \
    --lr_start 1e-8 \
    --lr_end 1 \
    --lr_iter 200 \
    --lr_advanced \
    --lr_step_mode exp \
    --lr_smooth_f 0.1 \
    --lr_diverge_th 3 \
    --output_dir ./lr_results
```

### 3. Run LR Finder directly with main.py

```bash
python main.py \
    --dataset imagenet1k \
    --batch_size 32 \
    --max_samples 1000 \
    --find_lr \
    --lr_advanced \
    --lr_start 1e-7 \
    --lr_end 10 \
    --lr_iter 100 \
    --lr_plot ./lr_finder_plot.png \
    --lr_step_mode exp \
    --lr_smooth_f 0.05 \
    --lr_diverge_th 5 \
    --epochs 1
```

## Parameters

### Dataset Options
- `--dataset`: Choose between `cifar100` and `imagenet1k`
- `--batch_size`: Batch size for training (smaller for LR finder, e.g., 32)
- `--streaming`: Use streaming for large datasets (default: True for ImageNet-1K)
- `--max_samples`: Maximum number of samples to use (useful for testing/debugging)

### LR Finder Parameters
- `--find_lr`: Enable LR finder mode
- `--lr_advanced`: Use advanced LR finder with more options
- `--lr_start`: Starting learning rate (default: 1e-7)
- `--lr_end`: Ending learning rate (default: 10)
- `--lr_iter`: Number of iterations to test (default: 100)
- `--lr_plot`: Path to save the LR finder plot
- `--lr_step_mode`: Step mode - 'exp' for exponential, 'linear' for linear (default: exp)
- `--lr_smooth_f`: Smoothing factor for loss (default: 0.05)
- `--lr_diverge_th`: Divergence threshold (default: 5)

### Other Parameters
- `--data_dir`: Directory containing the dataset
- `--no_cuda`: Disable CUDA if needed
- `--amp`: Enable automatic mixed precision

## Understanding the Results

### LR Finder Plot
The generated plot shows:
- X-axis: Learning rate (log scale)
- Y-axis: Loss
- Red dashed line: Suggested optimal learning rate

### Interpreting the Plot
1. **Steepest Descent**: The LR finder suggests the learning rate at the steepest descent point
2. **Loss Plateau**: Look for where the loss starts to plateau or increase
3. **Divergence**: The finder stops if loss diverges (becomes too high)

### Suggested Learning Rate
The finder will output a suggested learning rate like:
```
Suggested learning rate (steepest): 1.23e-02
```

## Example Workflow

1. **Find Optimal LR**:
   ```bash
   python run_lr_finder.py --dataset imagenet1k --batch_size 32
   ```

2. **Use Suggested LR for Training**:
   ```bash
   python main.py \
       --dataset imagenet1k \
       --batch_size 128 \
       --lr 1.23e-02 \
       --epochs 50 \
       --scheduler cosine
   ```

## Tips

1. **Batch Size**: Use smaller batch sizes (16-64) for LR finder to get more iterations
2. **Iterations**: 100-200 iterations usually sufficient for most datasets
3. **LR Range**: Start with 1e-7 to 10, adjust based on your model
4. **Memory**: LR finder uses more memory due to gradient accumulation

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Convergence**: Increase `--lr_iter` or adjust LR range
3. **Divergence**: Lower `--lr_end` or check data preprocessing

### Error Messages

- `"No data to plot"`: Run range_test first
- `"LR finder failed"`: Check data loading and model compatibility
- `"Stopping early due to divergence"`: Normal behavior, adjust LR range
- `"torch-lr-finder is not installed"`: Install with `pip install torch-lr-finder`

## Advanced Usage

### Custom LR Finder

You can also use the LR finder programmatically:

```python
from lr_finder import find_lr, find_lr_advanced, LRFinder
import torch.optim as optim

# Simple LR finder
suggested_lr, fig = find_lr(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    start_lr=1e-7,
    end_lr=10,
    num_iter=100,
    plot=True,
    save_path="simple_lr_plot.png"
)

# Advanced LR finder with more options
suggested_lr, fig = find_lr_advanced(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    start_lr=1e-7,
    end_lr=10,
    num_iter=100,
    step_mode="exp",
    smooth_f=0.05,
    diverge_th=5,
    plot=True,
    save_path="advanced_lr_plot.png"
)

# Direct usage of torch-lr-finder
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
lr_finder.plot()
suggested_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(min(lr_finder.history['loss']))]
lr_finder.reset()
```

## Dataset Support

### ImageNet1K (Streaming)
- **Classes**: 1000
- **Input Size**: 224x224
- **Transforms**: Standard ImageNet preprocessing
- **Loading**: Streaming from Hugging Face (no local download)
- **Memory**: Efficient streaming - only loads batches as needed
- **Max Samples**: Use `--max_samples` to limit dataset size for testing

### CIFAR-100
- **Classes**: 100
- **Input Size**: 32x32
- **Transforms**: CIFAR-specific preprocessing
- **Memory**: ~2GB GPU sufficient

## Performance Notes

- LR finder typically takes 5-15 minutes depending on dataset size
- ImageNet1K tiny dataset is recommended for faster iteration
- Full ImageNet1K can be used but will take longer
- Use smaller batch sizes for more stable LR finding
