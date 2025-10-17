# ImageNet-1K Learning Rate Finder

A PyTorch implementation for finding optimal learning rates for ImageNet-1K classification using Microsoft's ResNet-50 v1.5 architecture with offline data processing.

## Features

- **Microsoft ResNet-50 v1.5**: Official implementation based on [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)
- **Learning Rate Finder**: Integrated `torch-lr-finder` for optimal LR discovery
- **Offline Data Processing**: Uses local ImageNet-1K data files
- **1 Full Epoch Analysis**: Complete dataset pass for accurate LR finding
- **Detailed Logging**: Comprehensive logs and analysis of LR findings
- **Batch Size 256**: Optimized for ImageNet-1K training

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Learning Rate Finder](#learning-rate-finder)
- [Model Architecture](#model-architecture)
- [Data Requirements](#data-requirements)
- [Output Files](#output-files)
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
- `albumentations>=1.4` - Advanced image augmentation
- `torchsummary>=1.5` - Model architecture visualization

## Quick Start

### 1. Prepare ImageNet-1K Data

Ensure you have ImageNet-1K data files in the `./data` directory with the following structure:
```
./data/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ... (1000 class folders)
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ... (1000 class folders)
```

### 2. Run Learning Rate Finder

```bash
# Basic LR finder (1 full epoch, batch size 256)
uv run python main.py \
    --batch_size 256 \
    --find_lr \
    --lr_start 1e-07 \
    --lr_end 10 \
    --lr_iter 1000 \
    --lr_plot ./outputs/lr_finder_imagenet1k.png \
    --data_dir ./data \
    --epochs 1 \
    --no_plots

# Using the helper script
uv run python run_lr_finder.py \
    --batch_size 256 \
    --lr_start 1e-07 \
    --lr_end 10 \
    --lr_iter 1000 \
    --output_dir ./outputs
```

### 3. View Results

After completion, check the output files:
- `./outputs/lr_finder_imagenet1k.png` - LR vs Loss plot
- `./outputs/lr_finder_log.txt` - Detailed analysis log
- `./outputs/suggested_lr.json` - Machine-readable results

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

The LR Finder helps identify the optimal learning rate by running 1 full epoch on ImageNet-1K data and systematically testing different learning rates.

### How It Works

1. **Load ImageNet-1K Data**: Uses offline data from the `./data` directory
2. **Run 1 Full Epoch**: Processes the entire training dataset once
3. **Test LR Range**: Systematically tests learning rates from `1e-07` to `10`
4. **Record Loss vs LR**: For each batch, records the loss and current learning rate
5. **Generate Analysis**: Creates plot and detailed log of findings
6. **Suggest Optimal LR**: Identifies the learning rate with steepest loss descent

### Process Timeline

```
Start → Load Data → Run 1 Epoch (testing LRs) → Generate Plot → Save Results → Complete
```

### Key Features

- **1 Full Epoch**: Complete dataset pass for accurate analysis
- **Batch Size 256**: Optimized for ImageNet-1K training
- **Offline Data**: Uses local data files (no streaming)
- **Detailed Logging**: Comprehensive analysis and recommendations
- **Visual Output**: LR vs Loss plot for easy interpretation

### Usage

#### Basic LR Finder

```bash
uv run python main.py \
    --batch_size 256 \
    --find_lr \
    --lr_start 1e-07 \
    --lr_end 10 \
    --lr_iter 1000 \
    --lr_plot ./outputs/lr_finder_imagenet1k.png \
    --data_dir ./data \
    --epochs 1 \
    --no_plots
```

#### Using Helper Script

```bash
uv run python run_lr_finder.py \
    --batch_size 256 \
    --lr_start 1e-07 \
    --lr_end 10 \
    --lr_iter 1000 \
    --output_dir ./outputs
```

#### Advanced Options

```bash
uv run python main.py \
    --batch_size 256 \
    --find_lr \
    --lr_advanced \
    --lr_start 1e-08 \
    --lr_end 1 \
    --lr_iter 500 \
    --lr_step_mode exp \
    --lr_smooth_f 0.1 \
    --lr_diverge_th 3 \
    --lr_plot ./outputs/advanced_lr_finder.png \
    --data_dir ./data \
    --epochs 1 \
    --no_plots
```

### Interpreting Results

The LR finder generates:

1. **Plot**: `lr_finder_imagenet1k.png`
   - X-axis: Learning rate (log scale)
   - Y-axis: Training loss
   - Red dashed line: Suggested optimal learning rate

2. **Log File**: `lr_finder_log.txt`
   - Detailed configuration and analysis
   - Warnings for very low/high learning rates
   - Recommended next steps

3. **JSON File**: `suggested_lr.json`
   - Machine-readable results
   - Suggested learning rate value
   - Configuration metadata

**Guidelines:**
- Use the LR at the steepest descent (before loss plateaus)
- Typically 1/10th of the maximum LR where loss starts increasing
- Start training with suggested LR or slightly lower
- Monitor training loss for divergence

## Data Requirements

### ImageNet-1K Data Structure

The LR finder requires ImageNet-1K data to be organized in the following structure:

```
./data/
├── train/
│   ├── n01440764/          # Class 1
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   └── ...
│   ├── n01443537/          # Class 2
│   │   ├── n01443537_10007.JPEG
│   │   └── ...
│   └── ... (1000 classes total)
└── val/
    ├── n01440764/          # Class 1 validation
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    ├── n01443537/          # Class 2 validation
    │   └── ...
    └── ... (1000 classes total)
```

### Data Augmentation

**Training Transforms:**
- Resize to 256×256
- Random crop to 224×224
- Horizontal flip (50%)
- Color jitter (brightness, contrast, saturation, hue)
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Validation Transforms:**
- Resize to 256×256
- Center crop to 224×224
- ImageNet normalization

### Data Loading

- **Offline Processing**: Uses local data files (no streaming)
- **Batch Size**: 256 (optimized for ImageNet-1K)
- **Workers**: 4 (adjustable based on CPU cores)
- **Pin Memory**: Enabled for CUDA acceleration

## Output Files

After running the learning rate finder, the following files are generated:

### 1. LR Finder Plot (`lr_finder_imagenet1k.png`)

Visual representation of the learning rate analysis:
- **X-axis**: Learning rate (log scale from 1e-07 to 10)
- **Y-axis**: Training loss
- **Red dashed line**: Suggested optimal learning rate
- **Smooth curve**: Shows relationship between LR and loss

### 2. Detailed Log (`lr_finder_log.txt`)

Comprehensive text log containing:
- **Configuration**: All parameters used
- **Command executed**: Exact command that was run
- **Output**: Complete stdout and stderr from the process
- **Results**: Suggested learning rate and success status
- **Analysis**: Intelligent analysis of the suggested LR
- **Warnings**: Alerts for very low/high learning rates
- **Recommendations**: Next steps for training

### 3. JSON Results (`suggested_lr.json`)

Machine-readable results file:
```json
{
  "suggested_lr": 0.00123,
  "dataset": "imagenet1k",
  "batch_size": 256,
  "lr_finder_epochs": 1,
  "plot_file": "./outputs/lr_finder_imagenet1k.png",
  "log_file": "./outputs/lr_finder_log.txt",
  "timestamp": "2024-01-15T10:30:45"
}
```

### 4. Example Log Analysis

The log file includes intelligent analysis:

```
================================================================================
ANALYSIS
================================================================================
✅ Learning rate appears to be in a reasonable range

Recommended next steps:
1. Use learning rate: 1.23e-03
2. Monitor training loss and accuracy
3. Adjust learning rate if needed during training
4. Consider using learning rate scheduling
```

## Using Results for Training

After finding the optimal learning rate, you can use it for training your model:

### Example: Using Discovered Learning Rate

```bash
# After LR finder suggests: 0.00123
uv run python main.py \
    --batch_size 256 \
    --lr 0.00123 \
    --epochs 50 \
    --scheduler cosine \
    --data_dir ./data
```

### Learning Rate Scheduling

**Cosine Annealing** (Recommended):
```bash
--scheduler cosine
```

**Step LR**:
```bash
--scheduler step --step_size 30 --gamma 0.1
```

**OneCycle LR** (Fast convergence):
```bash
--scheduler onecycle
```

## Command Line Arguments

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | 256 | Batch size for LR finder |
| `--epochs` | int | 1 | Number of epochs (always 1 for LR finder) |
| `--data_dir` | str | ./data | ImageNet-1K data directory |
| `--max_samples` | int | None | Limit samples (for testing) |
| `--no_cuda` | flag | False | Disable CUDA |

### LR Finder Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--find_lr` | flag | False | Run learning rate finder |
| `--lr_advanced` | flag | False | Use advanced LR finder |
| `--lr_start` | float | 1e-7 | Starting learning rate |
| `--lr_end` | float | 10 | Ending learning rate |
| `--lr_iter` | int | 1000 | Number of iterations |
| `--lr_plot` | str | ./lr_finder_plot.png | Plot save path |
| `--lr_step_mode` | str | exp | Step mode: `exp` or `linear` |
| `--lr_smooth_f` | float | 0.05 | Smoothing factor |
| `--lr_diverge_th` | float | 5 | Divergence threshold |

### Output Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--no_plots` | flag | False | Disable other plots (recommended for LR finder) |

## Examples

### Example 1: Basic LR Finder

```bash
# Find optimal learning rate with default settings
uv run python main.py \
    --batch_size 256 \
    --find_lr \
    --lr_start 1e-07 \
    --lr_end 10 \
    --lr_iter 1000 \
    --lr_plot ./outputs/lr_finder_imagenet1k.png \
    --data_dir ./data \
    --epochs 1 \
    --no_plots
```

### Example 2: Using Helper Script

```bash
# Use the simplified helper script
uv run python run_lr_finder.py \
    --batch_size 256 \
    --lr_start 1e-07 \
    --lr_end 10 \
    --lr_iter 1000 \
    --output_dir ./outputs
```

### Example 3: Advanced LR Finder

```bash
# Advanced LR finder with custom parameters
uv run python main.py \
    --batch_size 256 \
    --find_lr \
    --lr_advanced \
    --lr_start 1e-08 \
    --lr_end 1 \
    --lr_iter 500 \
    --lr_step_mode exp \
    --lr_smooth_f 0.1 \
    --lr_diverge_th 3 \
    --lr_plot ./outputs/advanced_lr_finder.png \
    --data_dir ./data \
    --epochs 1 \
    --no_plots
```

### Example 4: Quick Test with Limited Data

```bash
# Test with limited samples for faster execution
uv run python main.py \
    --batch_size 128 \
    --find_lr \
    --max_samples 1000 \
    --lr_start 1e-07 \
    --lr_end 1 \
    --lr_iter 200 \
    --lr_plot ./outputs/test_lr_finder.png \
    --data_dir ./data \
    --epochs 1 \
    --no_plots
```

### Example 5: Custom Output Directory

```bash
# Save results to custom directory
uv run python main.py \
    --batch_size 256 \
    --find_lr \
    --lr_start 1e-07 \
    --lr_end 10 \
    --lr_iter 1000 \
    --lr_plot ./custom_results/my_lr_finder.png \
    --data_dir ./data \
    --epochs 1 \
    --no_plots

# Results will be saved to:
# - ./custom_results/my_lr_finder.png
# - ./custom_results/lr_finder_log.txt
# - ./custom_results/suggested_lr.json
```

## Project Structure

```
Imagenet1K/
├── main.py                    # Main script with LR finder
├── find_lr.py                 # LR finder helper script
├── run_lr_finder.py           # Simplified LR finder script
├── model_resnet50.py          # ResNet-50 v1.5 implementation
├── preprocess.py              # Data loading and augmentation
├── lr_finder.py               # LR finder implementation
├── train.py                   # Training and evaluation functions
├── pyproject.toml             # Project configuration
├── requirements.txt           # Python dependencies
├── data/                      # ImageNet-1K data directory
│   ├── train/                 # Training images (1000 classes)
│   └── val/                   # Validation images (1000 classes)
└── README.md                  # This file
```

## Performance Tips

### For Faster LR Finding

1. **Reduce Batch Size**: If memory limited, use `--batch_size 128`
2. **Limit Samples**: Use `--max_samples 1000` for quick testing
3. **Fewer Iterations**: Use `--lr_iter 500` for faster execution
4. **More Workers**: Adjust `--num_workers` based on CPU cores
5. **SSD Storage**: Faster data loading from SSD

### For Better Accuracy

1. **Full Dataset**: Use complete ImageNet-1K (no `--max_samples`)
2. **More Iterations**: Use `--lr_iter 1000` or higher
3. **Wider LR Range**: Test from `1e-08` to `10`
4. **Advanced Mode**: Use `--lr_advanced` for better analysis

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Reduce batch size
--batch_size 128

# Or limit samples
--max_samples 1000
```

**2. Data Not Found**
```bash
# Check data directory structure
ls -la ./data/train/
ls -la ./data/val/

# Ensure proper ImageNet-1K structure
```

**3. Slow Data Loading**
```bash
# Increase workers (adjust based on CPU cores)
--num_workers 8

# Use SSD storage for faster I/O
```

**4. LR Finder Takes Too Long**
```bash
# Reduce iterations
--lr_iter 500

# Use fewer samples
--max_samples 2000
```

**5. No Plot Generated**
```bash
# Check output directory exists
mkdir -p ./outputs

# Ensure write permissions
chmod 755 ./outputs
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

