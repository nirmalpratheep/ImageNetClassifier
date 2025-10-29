# PyTorch Lightning Training with TensorBoard

This implementation uses PyTorch Lightning for training with automatic TensorBoard logging and multi-GPU support.

## Features

- ✅ **TensorBoard Logging**: All metrics automatically logged to TensorBoard
  - Training loss (per-step and per-epoch)
  - Training accuracy (per-step and per-epoch)
  - Validation loss (per-epoch)
  - Validation accuracy (per-epoch)
  - Learning rate (per-epoch, via LearningRateMonitor)

- ✅ **Multi-GPU Support**: Automatic multi-GPU training with DDP
- ✅ **Mixed Precision**: Support for FP16/BF16 training
- ✅ **Checkpointing**: Automatic checkpoint saving with resume support
- ✅ **Custom Scheduler**: Three-phase LR scheduler preserved

## Installation

```bash
# Install dependencies (includes pytorch-lightning and tensorboard)
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python main_lightning.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.01
```

### With LR Finder

```bash
python main_lightning.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 50 \
    --find_lr \
    --lr_start 1e-7 \
    --lr_end 10
```

### Multi-GPU Training

```bash
# Automatic detection (uses all available GPUs)
python main_lightning.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 50

# Specify number of GPUs
python main_lightning.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 50 \
    --devices 4

# Use specific strategy
python main_lightning.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 50 \
    --strategy ddp
```

### Mixed Precision Training

```bash
python main_lightning.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 50 \
    --amp  # or --precision 16
```

### Resume from Checkpoint

```bash
python main_lightning.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 50 \
    --resume_from ./checkpoints/last.ckpt
```

## TensorBoard Viewing

After training starts, view metrics in TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir ./logs

# Or specify the exact experiment
tensorboard --logdir ./logs/imagenet_resnet50
```

Then open your browser to `http://localhost:6006`

## Metrics in TensorBoard

The following metrics are automatically logged:

- **train_loss**: Training loss (both step and epoch level)
- **train_acc**: Training accuracy (both step and epoch level) - values 0-1 (multiply by 100 for %)
- **val_loss**: Validation loss (epoch level)
- **val_acc**: Validation accuracy (epoch level) - values 0-1 (multiply by 100 for %)
- **learning_rate**: Learning rate (epoch level)

## Arguments

### Training
- `--batch_size`: Batch size per GPU (default: 256)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.01)
- `--momentum`: SGD momentum (default: 0.9)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--label_smoothing`: Label smoothing (default: 0.1)

### Scheduler
- `--scheduler`: Scheduler type - onecycle, cosine, step (default: onecycle)
- `--step_size`: Step size for StepLR (default: 15)
- `--gamma`: Gamma for StepLR (default: 0.1)

### Data
- `--data_dir`: Data directory (default: ./data)
- `--image_size`: Image size (default: 224)
- `--num_workers`: Data loader workers (default: 4)
- `--subset_size`: Limit number of classes (default: None)
- `--max_samples_per_class`: Max samples per class (default: None)

### Multi-GPU
- `--devices`: Number of GPUs (None = auto-detect)
- `--strategy`: Training strategy - auto, ddp, ddp_spawn (default: auto)
- `--accelerator`: Accelerator type (default: auto)

### Precision
- `--amp`: Enable mixed precision (FP16)
- `--precision`: Precision - 16, 32, bf16 (default: 32)

### Checkpointing
- `--checkpoint_dir`: Checkpoint directory (default: ./checkpoints)
- `--resume_from`: Resume from checkpoint path
- `--save_top_k`: Save top K checkpoints (default: 3)
- `--monitor`: Metric to monitor (default: val_acc)

### Logging
- `--log_dir`: TensorBoard log directory (default: ./logs)
- `--name`: Experiment name (default: imagenet_resnet50)
- `--version`: Experiment version (default: auto)

## Directory Structure

After training, you'll have:

```
./
├── checkpoints/
│   ├── epoch=49-val_acc=0.7234-val_loss=1.2345.ckpt  # Best checkpoints
│   └── last.ckpt  # Latest checkpoint
├── logs/
│   └── imagenet_resnet50/
│       └── version_0/
│           └── events.out.tfevents.*  # TensorBoard logs
```

## Key Differences from Original main.py

1. **Automatic Multi-GPU**: No manual DDP setup needed
2. **TensorBoard Integration**: All metrics automatically logged
3. **Checkpoint Management**: Automatic checkpoint saving with best model tracking
4. **Simplified Code**: Less boilerplate, more features
5. **Better Scalability**: Easy to scale to many GPUs/nodes

## Notes

- Batch size is per-GPU, total batch = batch_size × num_gpus
- Metrics sync automatically across GPUs (sync_dist=True)
- Learning rate is logged automatically via LearningRateMonitor callback
- Checkpoints include optimizer, scheduler, and model state

