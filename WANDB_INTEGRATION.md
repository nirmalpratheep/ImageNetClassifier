# Weights & Biases (wandb) Integration

This document describes the comprehensive Weights & Biases integration for the ImageNet-1K Learning Rate Finder and Training project.

## Overview

The project now includes full wandb integration for:

1. **LR Finder Experiments** - Track learning rate finding runs with plots and metrics
2. **Training Experiments** - Monitor full training runs with comprehensive metrics
3. **Helper Scripts** - Easy-to-use scripts with wandb support

## Installation

The project uses `uv` package manager with dependencies managed in `pyproject.toml`. Wandb is already included as a dependency.

```bash
# Install all dependencies including wandb
uv sync

# Or if using pip with requirements.txt
pip install -r requirements.txt
pip install wandb>=0.16.0
```

## Setup

First, login to wandb (one-time setup):

```bash
wandb login
```

## Features

### 🔍 LR Finder Integration

- **Automatic Logging**: LR vs Loss curves, suggested learning rates
- **Both Modes**: Basic and Advanced LR finder support  
- **Rich Metadata**: Model architecture, optimizer settings, device info
- **Visual Plots**: Interactive LR vs Loss visualizations

### 🚀 Training Integration

- **Comprehensive Metrics**: Train/val losses, accuracies, learning rates
- **Per-Batch Logging**: Optional batch-level metrics (configurable frequency)
- **Model Watching**: Automatic gradient and parameter tracking
- **Final Summaries**: Complete training statistics and best model metrics

### 🛠️ Helper Script Integration

- **find_lr.py**: Full wandb support with custom project/run names
- **train_with_lr.py**: Seamless training logging with LR from finder

## Usage

### Basic LR Finder with Wandb

```bash
python find_lr.py \
    --batch_size 256 \
    --lr_start 1e-7 \
    --lr_end 10 \
    --lr_iter 500 \
    --use_wandb \
    --wandb_project "my-imagenet-lr-experiments" \
    --wandb_run_name "lr_finder_resnet50_v1" \
    --wandb_tags lr_finder imagenet resnet50
```

### Training with Wandb

```bash
python main.py \
    --batch_size 256 \
    --epochs 10 \
    --lr 0.001 \
    --scheduler cosine \
    --use_wandb \
    --wandb_project "my-imagenet-training" \
    --wandb_run_name "resnet50_cosine_lr_001" \
    --wandb_tags training imagenet resnet50 cosine \
    --wandb_group "lr_comparison" \
    --wandb_notes "Testing cosine scheduler with suggested LR"
```

### Complete Workflow Example

1. **Find Optimal Learning Rate:**
```bash
python find_lr.py \
    --batch_size 256 \
    --use_wandb \
    --wandb_project "imagenet-experiments" \
    --wandb_tags lr_finder baseline
```

2. **Train with Found Learning Rate:**
```bash
python train_with_lr.py \
    --batch_size 256 \
    --auto_lr \
    --use_wandb \
    --wandb_project "imagenet-experiments" \
    --wandb_group "lr_finder_guided" \
    --wandb_tags training onecycle
```

## Command Line Arguments

### Core Wandb Arguments (Available in all scripts)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_wandb` | flag | False | Enable wandb logging |
| `--wandb_project` | str | varies | Wandb project name |
| `--wandb_run_name` | str | None | Custom run name (auto-generated if not set) |
| `--wandb_tags` | list | None | Space-separated tags for the run |

### Training-Specific Arguments (main.py, train_with_lr.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--wandb_group` | str | None | Group related runs together |
| `--wandb_notes` | str | None | Description/notes for the run |

## Logged Metrics

### LR Finder Experiments

**Basic Metrics:**
- `suggested_lr` - The recommended learning rate
- `min_loss` - Minimum loss encountered
- `max_loss` - Maximum loss encountered
- `lr_range_start` - Starting learning rate
- `lr_range_end` - Ending learning rate
- `total_iterations` - Number of iterations completed

**Advanced Metrics (Advanced LR Finder only):**
- `min_loss_lr` - LR at minimum loss point
- `steepest_descent_lr` - LR at steepest descent
- `step_mode` - Linear or exponential stepping
- `smooth_f` - Smoothing factor applied
- `diverge_th` - Divergence threshold used

**Plots:**
- `lr_vs_loss_plot` - Interactive LR vs Loss curve
- `lr_finder_matplotlib_plot` - Static matplotlib plot

### Training Experiments

**Per-Epoch Metrics:**
- `train/loss` - Training loss
- `train/accuracy` - Training accuracy
- `val/loss` - Validation loss  
- `val/accuracy` - Validation accuracy
- `learning_rate` - Current learning rate
- `best_val_accuracy` - Best validation accuracy so far

**Per-Batch Metrics (optional, every 100 batches):**
- `batch/loss` - Batch training loss
- `batch/accuracy` - Batch training accuracy
- `batch/learning_rate` - Current learning rate
- `batch/batch_idx` - Batch index
- `batch/epoch` - Current epoch

**Final Summary:**
- `final/train_loss` - Final training loss
- `final/train_accuracy` - Final training accuracy
- `final/val_loss` - Final validation loss
- `final/val_accuracy` - Final validation accuracy
- `final/best_val_accuracy` - Best validation accuracy achieved
- `final/total_epochs` - Total epochs completed

**Model Monitoring:**
- Automatic gradient tracking (every 100 steps)
- Parameter histograms
- Model architecture logging

## Project Structure

### Default Project Names

- **LR Finder**: `"imagenet-lr-finder"`
- **Training**: `"imagenet-classification"`

### Auto-Generated Run Names

- **LR Finder**: `lr_finder_basic_1e-07_1e+01` or `lr_finder_advanced_1e-08_1e+00`
- **Training**: `imagenet_resnet50_cosine_lr_1e-03_bs_256`

### Automatic Tags

- **LR Finder**: `["lr_finder", "basic"/"advanced", "imagenet"]`
- **Training**: `["imagenet", "resnet50", "training", scheduler_type]`
- **Mixed Precision**: `"mixed_precision"` (when `--amp` is enabled)

## Best Practices

### 🎯 Experiment Organization

1. **Use Consistent Project Names:**
   ```bash
   --wandb_project "imagenet-resnet50-experiments"
   ```

2. **Group Related Runs:**
   ```bash
   --wandb_group "lr_schedule_comparison"
   ```

3. **Use Descriptive Tags:**
   ```bash
   --wandb_tags baseline optimized mixed_precision
   ```

4. **Add Meaningful Notes:**
   ```bash
   --wandb_notes "Testing impact of label smoothing on convergence"
   ```

### 📊 Monitoring Strategy

1. **LR Finding Phase:**
   - Run LR finder with wandb to compare different LR ranges
   - Use tags to distinguish different model architectures
   - Compare suggested LRs across different batch sizes

2. **Training Phase:**
   - Enable batch-level logging for debugging (default: every 100 batches)
   - Use groups to compare different hyperparameter settings
   - Monitor gradient norms and parameter distributions

3. **Analysis Phase:**
   - Compare training curves across different schedulers
   - Analyze convergence patterns
   - Identify optimal hyperparameter combinations

### 🔧 Performance Tips

1. **Batch Logging Frequency:**
   - Default: Every 100 batches (good balance)
   - For debugging: Every 10-50 batches
   - For long runs: Every 200-500 batches

2. **Model Watching:**
   - Automatically enabled with reasonable frequency (100 steps)
   - Disabled for graph logging to avoid memory issues
   - Tracks gradients and parameters for debugging

## Troubleshooting

### Common Issues

1. **Wandb Not Available:**
   ```
   ⚠️  Warning: Wandb requested but not available. Install with: pip install wandb
   ```
   **Solution:** Install wandb: `pip install wandb>=0.16.0`

2. **Authentication Error:**
   ```
   wandb: ERROR authentication failed
   ```
   **Solution:** Run `wandb login` and enter your API key

3. **Project Not Found:**
   **Solution:** Check project name spelling or create it on wandb web interface

4. **Too Many Logs:**
   **Solution:** Increase batch logging frequency with `log_freq` parameter

### Configuration Check

Test your wandb setup:

```bash
python -c "import wandb; print('Wandb version:', wandb.__version__)"
wandb --version
```

## Examples

### 1. Quick LR Finder Test

```bash
python find_lr.py \
    --batch_size 128 \
    --lr_iter 200 \
    --max_samples 1000 \
    --use_wandb \
    --wandb_run_name "quick_lr_test"
```

### 2. Production LR Finding

```bash
python find_lr.py \
    --batch_size 256 \
    --lr_start 1e-8 \
    --lr_end 1 \
    --lr_iter 1000 \
    --lr_advanced \
    --use_wandb \
    --wandb_project "production-imagenet" \
    --wandb_group "lr_optimization" \
    --wandb_tags production lr_finder advanced
```

### 3. Hyperparameter Sweep Training

```bash
# Run 1: Cosine scheduler
python main.py \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.001 \
    --scheduler cosine \
    --use_wandb \
    --wandb_project "scheduler-comparison" \
    --wandb_group "lr_001_comparison" \
    --wandb_run_name "cosine_lr_001" \
    --wandb_tags cosine scheduler_comparison

# Run 2: OneCycle scheduler  
python main.py \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.001 \
    --scheduler onecycle \
    --use_wandb \
    --wandb_project "scheduler-comparison" \
    --wandb_group "lr_001_comparison" \
    --wandb_run_name "onecycle_lr_001" \
    --wandb_tags onecycle scheduler_comparison
```

## Integration Summary

✅ **Complete Integration**: All major components support wandb logging
✅ **Backward Compatible**: Wandb is optional, code works without it
✅ **Rich Metrics**: Comprehensive logging of all important training metrics
✅ **Visual Plots**: Interactive LR finder plots and training curves  
✅ **Easy Configuration**: Simple command-line arguments
✅ **Best Practices**: Automatic tagging, grouping, and organization
✅ **Production Ready**: Robust error handling and performance optimization

The wandb integration provides complete experiment tracking for both learning rate finding and ImageNet training, making it easy to compare runs, analyze performance, and reproduce results.
