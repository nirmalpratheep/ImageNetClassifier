# ImageNet-1K Training with Learning Rate Finder

A PyTorch implementation for training ResNet-50 on ImageNet-1K with automatic learning rate finding using `main.py`.

## Quick Start

### 1. Prepare Your Data

Ensure you have ImageNet-1K data in the following structure:

```
./data/
├── train/
│   ├── n01440764/        # Class 1
│   ├── n01443537/        # Class 2
│   └── ... (1000 classes)
└── val/
    ├── n01440764/        # Class 1 validation
    ├── n01443537/        # Class 2 validation
    └── ... (1000 classes)
```

### 2. Run Training with LR Finder

```bash
uv run python main.py \
    --batch_size 256 \
    --find_lr \
    --lr_start 1e-02 \
    --lr_end 1 \
    --lr_iter 1000 \
    --lr_plot ./outputs/lr_finder_imagenet1k.png \
    --data_dir ./data \
    --epochs 50
```

### 3. Run Training with Found LR

After finding the optimal LR, train for longer:

```bash
uv run python main.py \
    --batch_size 256 \
    --lr 0.016681 \
    --data_dir ./data \
    --epochs 50 \
    --scheduler onecycle
```

## What This Does

1. **Finds Optimal Learning Rate**: Runs LR finder to suggest the best learning rate
2. **Trains the Model**: Uses the suggested LR with OneCycle scheduler
3. **Three-Phase Schedule**: 
   - Phase 1 (40%): LR increases from base → max
   - Phase 2 (40%): LR decreases from max → base
   - Phase 3 (20%): LR decreases from base → min

## Installation

```bash
# Install dependencies
uv sync
```

## Key Features

- **Automatic LR Finding**: Discover optimal learning rates using torch-lr-finder
- **OneCycle Scheduler**: Custom three-phase schedule (40%/40%/20%)
- **ResNet-50 Architecture**: Microsoft ResNet-50 v1.5
- **Progress Bars**: Real-time training progress with tqdm
- **Validation Metrics**: Prints train/val accuracy and loss each epoch

## Project Structure

```
├── main.py              # Main training script
├── dataset_loader.py    # Data loading
├── train.py             # Training functions
├── model_resnet50.py    # ResNet-50 model
├── lr_finder.py         # LR finder implementation
└── data/                # Your ImageNet data
```

## Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch_size` | Batch size | 256 |
| `--epochs` | Number of epochs | 50 |
| `--lr` | Learning rate (if not using finder) | 0.01 |
| `--find_lr` | Run LR finder first | - |
| `--scheduler` | LR scheduler (onecycle, cosine, step) | onecycle |
| `--max_samples_per_class` | Limit samples per class | None |
| `--data_dir` | Data directory | ./data |

## Output Files

- `./outputs/lr_finder_imagenet1k.png` - LR finder plot
- `./outputs/lr_finder_imagenet1k_summary.txt` - LR finder summary
- `./snapshots/` - Model checkpoints
- `./plots/` - Training visualizations (if enabled)

## Examples

### Basic Training
```bash
uv run python main.py --epochs 3
```

### With LR Finder
```bash
uv run python main.py --find_lr --epochs 3
```

### Limit Samples (for testing)
```bash
uv run python main.py --max_samples_per_class 100 --epochs 3
```

### Save Best Model
```bash
uv run python main.py --epochs 50 --save_best
```

## Troubleshooting

**Out of Memory?**
```bash
--batch_size 128
```

**Too Slow?**
```bash
--max_samples_per_class 100  # Limit samples
--num_workers 2
```

**No CUDA?**
```bash
--no_cuda
```

## License

Apache 2.0
