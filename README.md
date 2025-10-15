# ImageNet-1K Classification with LR Finder

A PyTorch implementation for ImageNet-1K classification using Microsoft's ResNet-50 v1.5 architecture with integrated Learning Rate Finder and streaming dataset support.

## Features

- **Microsoft ResNet-50 v1.5**: Official implementation based on [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)
- **Learning Rate Finder**: Integrated `torch-lr-finder` for optimal LR discovery
- **Streaming Dataset**: Memory-efficient loading via Hugging Face datasets
- **Flexible Training**: Support for both ImageNet-1K and CIFAR-100
- **Pretrained Weights**: Optional loading of Microsoft's pretrained ResNet-50 weights
- **Modern PyTorch**: Mixed precision training, gradient clipping, advanced schedulers

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Learning Rate Finder](#learning-rate-finder)
- [Dataset Support](#dataset-support)
- [Training](#training)
- [Command Line Arguments](#command-line-arguments)
- [Examples](#examples)
- [Project Structure](#project-structure)

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (optional, CPU training supported)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Required Packages

- `torch>=2.2` - PyTorch deep learning framework
- `torchvision>=0.17` - Computer vision utilities
- `torch-lr-finder>=0.2.1` - Learning rate finder
- `datasets>=2.14.0` - Hugging Face datasets for streaming
- `transformers>=4.30.0` - For pretrained model loading
- `albumentations>=1.4` - Advanced image augmentation
- `torchsummary>=1.5` - Model architecture visualization

## Quick Start

### 1. Find Optimal Learning Rate

```bash
# Run LR finder on ImageNet-1K (streaming, limited samples)
uv run python run_lr_finder.py --dataset imagenet1k --batch_size 32 --max_samples 1000

# With advanced options
uv run python run_lr_finder.py \
    --dataset imagenet1k \
    --batch_size 32 \
    --max_samples 2000 \
    --lr_advanced \
    --lr_start 1e-7 \
    --lr_end 10 \
    --output_dir ./lr_results
```

### 2. Train the Model

```bash
# Train with discovered learning rate
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 128 \
    --lr 0.01 \
    --epochs 50 \
    --scheduler cosine

# With pretrained weights
uv run python main.py \
    --dataset imagenet1k \
    --use_pretrained \
    --batch_size 128 \
    --lr 0.001 \
    --epochs 20
```

### 3. Test the Implementation

```bash
# Run comprehensive tests
uv run python test_lr_finder.py

# Test streaming dataset
uv run python test_streaming.py
```

## Model Architecture

### ResNet-50 v1.5

This implementation uses Microsoft's ResNet-50 v1.5 architecture, which differs from the original ResNet-50:

**Key Differences:**
- **v1.5 Improvement**: Stride=2 in the 3x3 convolution instead of the 1x1 convolution in bottleneck blocks
- **Better Accuracy**: ~0.5% higher top-1 accuracy compared to v1
- **Trade-off**: ~5% slower inference compared to v1

**Architecture Details:**
- **Input Size**: 224×224 for ImageNet, 32×32 for CIFAR
- **Total Parameters**: 25,557,032 (25.5M)
- **Bottleneck Blocks**: [3, 4, 6, 3] configuration
- **Output Classes**: 1000 (ImageNet) or 100 (CIFAR-100)

**Model Summary:**
```
----------------------------------------------------------------
Total params: 25,557,032
Trainable params: 25,557,032
Non-trainable params: 0
Input size (MB): 0.57
Forward/backward pass size (MB): 219.37
Params size (MB): 97.49
Estimated Total Size (MB): 317.44
----------------------------------------------------------------
```

### Pretrained Weights

Load pretrained weights from Microsoft's ResNet-50 model:

```python
from model_resnet50 import load_pretrained_resnet50

model = load_pretrained_resnet50(
    device=device,
    num_classes=1000,
    input_size=224
)
```

Or via command line:
```bash
uv run python main.py --dataset imagenet1k --use_pretrained
```

## Learning Rate Finder

The LR Finder helps identify the optimal learning rate by:
1. Starting with a very small learning rate
2. Exponentially increasing it during training
3. Monitoring the loss to find the steepest descent point
4. Suggesting the optimal learning rate

### How It Works

Based on the [torch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder) library:

1. **Initialization**: Creates a copy of model and optimizer state
2. **Range Test**: Trains for N iterations with exponentially increasing LR
3. **Analysis**: Calculates gradients and finds the steepest descent point
4. **Suggestion**: Returns the LR at the point of fastest learning
5. **Reset**: Restores original model and optimizer state

### Usage

#### Simple LR Finder

```bash
uv run python run_lr_finder.py --dataset imagenet1k --max_samples 1000
```

#### Advanced LR Finder

```bash
uv run python run_lr_finder.py \
    --dataset imagenet1k \
    --batch_size 64 \
    --max_samples 2000 \
    --lr_advanced \
    --lr_start 1e-8 \
    --lr_end 1 \
    --lr_iter 200 \
    --lr_step_mode exp \
    --lr_smooth_f 0.1 \
    --lr_diverge_th 3
```

#### Programmatic Usage

```python
from lr_finder import find_lr

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
    save_path="lr_finder_plot.png"
)

print(f"Suggested LR: {suggested_lr}")
```

### Interpreting Results

The LR finder generates a plot showing:
- **X-axis**: Learning rate (log scale)
- **Y-axis**: Loss
- **Red dashed line**: Suggested optimal learning rate

**Guidelines:**
- Use the LR at the steepest descent (before the loss plateaus)
- Typically 1/10th of the maximum LR where loss starts to increase
- Start training with the suggested LR or slightly lower

## Dataset Support

### ImageNet-1K (Streaming)

**Features:**
- **No Local Download**: Streams data from Hugging Face
- **Memory Efficient**: Loads batches on-demand
- **1000 Classes**: Full ImageNet-1K classification
- **1.2M Training Images**: Complete ImageNet training set
- **50K Validation Images**: Full validation set

**Usage:**
```python
from preprocess import get_data_loaders

train_loader, test_loader = get_data_loaders(
    batch_size=128,
    dataset_name="imagenet1k",
    streaming=True,
    max_samples=None  # Use full dataset
)
```

**Limited Samples (for testing):**
```bash
# Use only 1000 samples
uv run python main.py --dataset imagenet1k --max_samples 1000
```

**Note**: You need Hugging Face credentials to access ImageNet-1K. Set your token:
```bash
export HF_TOKEN=your_huggingface_token
```

### CIFAR-100

**Features:**
- **100 Classes**: 100 fine-grained categories
- **32×32 Images**: Smaller input size
- **50K Training**: 500 images per class
- **10K Testing**: 100 images per class
- **Local Download**: Automatic via torchvision

**Usage:**
```bash
uv run python main.py --dataset cifar100 --batch_size 128
```

### Data Augmentation

**ImageNet Transforms:**
- Resize to 256×256
- Random crop to 224×224
- Horizontal flip (50%)
- Color jitter (brightness, contrast, saturation)
- ImageNet normalization

**CIFAR-100 Transforms:**
- Pad to 36×36
- Random crop to 32×32
- Horizontal flip (50%)
- ShiftScaleRotate
- Coarse dropout (cutout)
- Color jitter

## Training

### Basic Training

```bash
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 128 \
    --epochs 50 \
    --lr 0.1 \
    --scheduler cosine
```

### Advanced Training

```bash
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 256 \
    --epochs 100 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --scheduler onecycle \
    --amp \
    --max_grad_norm 1.0 \
    --use_pretrained \
    --snapshot_freq 10 \
    --save_best
```

### Learning Rate Schedulers

**1. Cosine Annealing** (Recommended for standard training)
```bash
--scheduler cosine
```
- Smoothly decreases LR following a cosine curve
- Good for long training runs
- Simple and effective

**2. Step LR**
```bash
--scheduler step --step_size 30 --gamma 0.1
```
- Decreases LR by factor at fixed intervals
- Traditional approach
- Predictable behavior

**3. OneCycle LR** (Recommended for fast convergence)
```bash
--scheduler onecycle
```
- Based on PyTorch's official [OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
- Implements "Super-Convergence" technique from the paper
- Two-phase policy (warmup + annealing) or optional three-phase
- Steps per batch (not per epoch) for fine-grained control
- Cycles momentum inversely to learning rate

**OneCycleLR Advanced Options:**
```bash
uv run python main.py \
    --dataset imagenet1k \
    --scheduler onecycle \
    --lr 0.1 \
    --onecycle_pct_start 0.3 \
    --onecycle_div_factor 25.0 \
    --onecycle_final_div_factor 10000.0 \
    --onecycle_anneal_strategy cos \
    --onecycle_three_phase
```

**OneCycleLR Parameters:**
- `--onecycle_pct_start`: Warmup phase percentage (default: 0.3 = 30%)
- `--onecycle_div_factor`: Initial LR divisor (default: 25.0)
- `--onecycle_final_div_factor`: Final LR divisor (default: 10000.0)
- `--onecycle_anneal_strategy`: 'cos' or 'linear' (default: cos)
- `--onecycle_three_phase`: Enable three-phase schedule

**How OneCycleLR Works:**

Based on the [PyTorch OneCycleLR documentation](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html):

1. **Phase 1 (Warmup)**: LR increases from `initial_lr` to `max_lr`
   - Duration: `pct_start * total_steps` (default: 30% of training)
   - Momentum decreases from `max_momentum` to `base_momentum`

2. **Phase 2 (Annealing)**: LR decreases from `max_lr` to `min_lr`
   - Duration: Remaining steps
   - Momentum increases from `base_momentum` to `max_momentum`
   - Uses cosine or linear annealing

3. **Phase 3 (Optional)**: Further annealing to very low LR
   - Only if `three_phase=True`
   - Follows original paper more closely

**Learning Rate Schedule:**
- Initial LR = `max_lr / div_factor` (e.g., 0.1 / 25 = 0.004)
- Max LR = `--lr` parameter (e.g., 0.1)
- Min LR = `initial_lr / final_div_factor` (e.g., 0.004 / 10000 = 0.0000004)

**When to Use:**
- Fast convergence needed
- Limited training time
- Already found optimal max LR with LR finder
- Training from scratch (not fine-tuning)

### Mixed Precision Training

Enable automatic mixed precision (AMP) for faster training:
```bash
uv run python main.py --dataset imagenet1k --amp
```

**Benefits:**
- 2-3× faster training
- Reduced memory usage
- Same accuracy as FP32

## Command Line Arguments

### Dataset Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | cifar100 | Dataset: `cifar100` or `imagenet1k` |
| `--data_dir` | str | ./data | Data directory |
| `--streaming` | flag | True | Use streaming for large datasets |
| `--max_samples` | int | None | Limit samples (for testing) |

### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | resnet34 | Model architecture |
| `--use_pretrained` | flag | False | Load pretrained ResNet-50 weights |

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | 128 | Batch size |
| `--epochs` | int | 50 | Number of epochs |
| `--lr` | float | 0.1 | Learning rate |
| `--momentum` | float | 0.9 | SGD momentum |
| `--weight_decay` | float | 1e-4 | Weight decay |
| `--scheduler` | str | cosine | LR scheduler: `cosine`, `step`, `onecycle` |
| `--amp` | flag | False | Enable mixed precision |
| `--max_grad_norm` | float | 1.0 | Gradient clipping threshold |
| `--num_workers` | int | 4 | DataLoader workers |

### OneCycleLR Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--onecycle_pct_start` | float | 0.3 | Warmup phase percentage |
| `--onecycle_div_factor` | float | 25.0 | Initial LR = max_lr / div_factor |
| `--onecycle_final_div_factor` | float | 10000.0 | Min LR = initial_lr / final_div_factor |
| `--onecycle_anneal_strategy` | str | cos | Annealing: `cos` or `linear` |
| `--onecycle_three_phase` | flag | False | Use three-phase schedule |

### LR Finder Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--find_lr` | flag | False | Run LR finder |
| `--lr_advanced` | flag | False | Use advanced LR finder |
| `--lr_start` | float | 1e-7 | Starting LR |
| `--lr_end` | float | 10 | Ending LR |
| `--lr_iter` | int | 100 | Number of iterations |
| `--lr_plot` | str | ./lr_finder_plot.png | Plot save path |
| `--lr_step_mode` | str | exp | Step mode: `exp` or `linear` |
| `--lr_smooth_f` | float | 0.05 | Smoothing factor |
| `--lr_diverge_th` | float | 5 | Divergence threshold |

### Snapshot Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--snapshot_dir` | str | ./snapshots | Snapshot directory |
| `--snapshot_freq` | int | 5 | Save every N epochs |
| `--save_best` | flag | False | Save only best models |
| `--resume_from` | str | None | Resume from snapshot |

## Examples

### Example 1: Find LR and Train with OneCycleLR

```bash
# Step 1: Find optimal learning rate
uv run python run_lr_finder.py \
    --dataset imagenet1k \
    --batch_size 32 \
    --max_samples 1000 \
    --output_dir ./lr_results

# Step 2: Train with discovered LR using OneCycleLR (e.g., 0.1)
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 128 \
    --lr 0.1 \
    --epochs 50 \
    --scheduler onecycle \
    --onecycle_pct_start 0.3 \
    --onecycle_anneal_strategy cos \
    --amp \
    --save_best
```

### Example 2: Fine-tune Pretrained Model

```bash
# Fine-tune with lower learning rate
uv run python main.py \
    --dataset imagenet1k \
    --use_pretrained \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 20 \
    --scheduler cosine \
    --weight_decay 1e-5
```

### Example 3: Resume Training

```bash
# Resume from checkpoint
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 128 \
    --resume_from ./snapshots/resnet34_epoch_25.pth \
    --epochs 100
```

### Example 4: Quick Test with Limited Data

```bash
# Test with 100 samples
uv run python main.py \
    --dataset imagenet1k \
    --max_samples 100 \
    --batch_size 16 \
    --epochs 5 \
    --find_lr
```

### Example 5: Complete Workflow - LR Finder + OneCycleLR

```bash
# Step 1: Find optimal max LR
uv run python run_lr_finder.py \
    --dataset imagenet1k \
    --batch_size 32 \
    --max_samples 2000 \
    --output_dir ./lr_results

# Output: Suggested learning rate: 1.23e-01

# Step 2: Train with OneCycleLR using discovered max LR
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 256 \
    --lr 0.123 \
    --epochs 100 \
    --scheduler onecycle \
    --onecycle_pct_start 0.3 \
    --onecycle_div_factor 25.0 \
    --onecycle_final_div_factor 10000.0 \
    --onecycle_anneal_strategy cos \
    --amp \
    --save_best \
    --snapshot_dir ./checkpoints

# This will train with:
# - Initial LR: 0.00492 (0.123 / 25)
# - Max LR: 0.123 (discovered from LR finder)
# - Min LR: 0.000000492 (0.00492 / 10000)
# - 30% warmup, 70% annealing with cosine strategy
```

### Example 6: Three-Phase OneCycleLR (Original Paper)

```bash
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 128 \
    --lr 0.1 \
    --epochs 90 \
    --scheduler onecycle \
    --onecycle_three_phase \
    --onecycle_pct_start 0.45 \
    --amp
```

## Project Structure

```
Imagenet1K/
├── main.py                    # Main training script
├── train.py                   # Training and evaluation functions
├── model_resnet50.py          # ResNet-50 v1.5 implementation
├── preprocess.py              # Data loading and augmentation
├── lr_finder.py               # LR finder implementation
├── run_lr_finder.py           # LR finder helper script
├── test_lr_finder.py          # LR finder tests
├── test_streaming.py          # Streaming dataset tests
├── example_lr_finder.py       # LR finder examples
├── pyproject.toml             # Project configuration
├── requirements.txt           # Python dependencies
├── LR_FINDER_USAGE.md         # Detailed LR finder guide
└── README.md                  # This file
```

## Performance Tips

### For Faster Training

1. **Use Mixed Precision**: `--amp` (2-3× speedup)
2. **Increase Batch Size**: Limited by GPU memory
3. **More Workers**: `--num_workers 8` (adjust based on CPU)
4. **Pin Memory**: Automatically enabled for CUDA
5. **Streaming Dataset**: Reduces I/O overhead

### For Better Accuracy

1. **Find Optimal LR**: Use LR finder before training
2. **Use Pretrained Weights**: `--use_pretrained`
3. **Longer Training**: More epochs with cosine scheduler
4. **Label Smoothing**: Built-in (0.1)
5. **Data Augmentation**: Already optimized

### Memory Optimization

1. **Smaller Batch Size**: Reduce if OOM
2. **Gradient Accumulation**: Simulate larger batches
3. **Streaming Dataset**: No full dataset in memory
4. **Mixed Precision**: Reduces memory usage

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Reduce batch size
--batch_size 32

# Or limit samples
--max_samples 1000
```

**2. Slow Data Loading**
```bash
# Increase workers (adjust based on CPU cores)
--num_workers 8

# Use streaming
--streaming
```

**3. Hugging Face Authentication Error**
```bash
# Set your HF token
export HF_TOKEN=your_token

# Or login
huggingface-cli login
```

**4. LR Finder Takes Too Long**
```bash
# Reduce iterations
--lr_iter 50

# Use fewer samples
--max_samples 500
```

## Citation

If you use this implementation, please cite:

**ResNet Paper:**
```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

**OneCycleLR / Super-Convergence Paper:**
```bibtex
@article{smith2018super,
  title={Super-convergence: Very fast training of neural networks using large learning rates},
  author={Smith, Leslie N and Topin, Nicholay},
  journal={arXiv preprint arXiv:1708.07120},
  year={2018}
}
```

**Microsoft ResNet-50:**
- Model: [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)
- License: Apache 2.0

**torch-lr-finder:**
```bibtex
@misc{pytorch-lr-finder,
  author = {David Silva},
  title = {pytorch-lr-finder},
  year = {2017},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/davidtvs/pytorch-lr-finder}}
}
```

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- Microsoft for the ResNet-50 v1.5 implementation
- Hugging Face for datasets and transformers
- David Silva for torch-lr-finder
- PyTorch team for the deep learning framework

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This implementation prioritizes correctness and educational value. For production use, consider additional optimizations and safety checks.

