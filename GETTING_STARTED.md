# Getting Started Guide

Quick start guide for ImageNet-1K classification with LR Finder and OneCycleLR.

## Prerequisites

- Python 3.12+
- GPU with CUDA support (recommended, but CPU works)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
uv run python test_lr_finder.py
```

Expected output: All tests PASSED ✓

## Complete Training Workflow

### Step 1: Find Optimal Learning Rate

Run the LR finder on a subset of ImageNet-1K:

```bash
uv run python run_lr_finder.py \
    --dataset imagenet1k \
    --batch_size 32 \
    --max_samples 1000 \
    --output_dir ./lr_results
```

**What this does:**
- Loads 1000 ImageNet samples (streaming, no full download)
- Tests learning rates from 1e-7 to 10
- Runs 100 iterations
- Saves plot to `./lr_results/lr_finder_imagenet1k.png`
- Suggests optimal learning rate

**Example output:**
```
Suggested learning rate: 8.35e-02
LR finder plot saved to: ./lr_results/lr_finder_imagenet1k.png
```

### Step 2: Examine the LR Finder Plot

Open `./lr_results/lr_finder_imagenet1k.png` and look for:
- **Steepest descent**: Where the loss drops fastest (red line)
- **Before plateau**: Use LR before loss starts to plateau
- **Suggested LR**: Marked with red dashed line

### Step 3: Train with OneCycleLR

Use the discovered learning rate as `max_lr` for OneCycleLR:

```bash
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 128 \
    --lr 0.0835 \
    --epochs 50 \
    --scheduler onecycle \
    --onecycle_pct_start 0.3 \
    --onecycle_anneal_strategy cos \
    --amp \
    --save_best \
    --snapshot_dir ./models
```

**What this does:**
- Trains ResNet-50 on ImageNet-1K (streaming)
- Uses OneCycleLR scheduler:
  - Initial LR: 0.00334 (0.0835 / 25)
  - Max LR: 0.0835 (discovered from LR finder)
  - Min LR: 0.000000334 (0.00334 / 10000)
- 30% warmup, 70% annealing
- Mixed precision for faster training
- Saves best models to `./models/`

### Step 4: Monitor Training

Watch for:
- **Epoch 1-15**: LR increasing, fast accuracy gains
- **Epoch 15**: LR reaches max (0.0835)
- **Epoch 15-50**: LR decreasing, fine-tuning
- **Final epoch**: LR very low, final adjustments

## Quick Examples

### Example 1: Fast Test (5 minutes)

```bash
uv run python main.py \
    --dataset imagenet1k \
    --max_samples 500 \
    --batch_size 32 \
    --epochs 5 \
    --scheduler onecycle \
    --lr 0.1
```

### Example 2: LR Finder Only

```bash
uv run python run_lr_finder.py \
    --dataset imagenet1k \
    --max_samples 1000 \
    --batch_size 32
```

### Example 3: CIFAR-100 Training

```bash
# Find LR for CIFAR-100
uv run python run_lr_finder.py \
    --dataset cifar100 \
    --batch_size 128

# Train
uv run python main.py \
    --dataset cifar100 \
    --batch_size 128 \
    --lr 0.1 \
    --epochs 100 \
    --scheduler onecycle
```

### Example 4: Pretrained Model Fine-tuning

```bash
uv run python main.py \
    --dataset imagenet1k \
    --use_pretrained \
    --batch_size 64 \
    --lr 0.01 \
    --epochs 20 \
    --scheduler cosine
```

## Understanding the Output

### LR Finder Output

```
[OneCycleLR] PyTorch Official Implementation:
   - Max LR: 0.0835
   - Epochs: 50
   - Steps per epoch: 1563
   - Total steps: 78150
   - Initial LR: 0.003340 (max_lr / 25.0)
   - Min LR: 0.00000033 (initial_lr / 10000.0)
   - Anneal strategy: cos
   - Percent start: 30% (warmup phase)
```

This means:
- Training starts at LR = 0.003340
- LR increases to 0.0835 over first 30% (23,445 steps)
- LR decreases to 0.00000033 over remaining 70%
- Cosine annealing curve for smooth transitions

### Training Output

```
Epoch 1/50
Learning Rate: 0.003340
Train Loss: 6.8234 | Train Acc: 0.12% | Test Loss: 6.7891 | Test Acc: 0.20%

Epoch 15/50
Learning Rate: 0.083500  # Peak LR
Train Loss: 4.2156 | Train Acc: 25.34% | Test Loss: 4.1234 | Test Acc: 28.56%

Epoch 50/50
Learning Rate: 0.000000  # Minimum LR
Train Loss: 1.2345 | Train Acc: 68.90% | Test Loss: 1.4567 | Test Acc: 65.23%
```

## Troubleshooting

### Issue: "Invalid credentials" for ImageNet-1K

**Solution**: The code falls back to dummy dataset automatically. To use real ImageNet:

```bash
# Login to Hugging Face
huggingface-cli login

# Or set token
export HF_TOKEN=your_huggingface_token
```

### Issue: Out of memory

**Solution**: Reduce batch size or limit samples

```bash
--batch_size 32 --max_samples 500
```

### Issue: LR finder takes too long

**Solution**: Reduce iterations or samples

```bash
--lr_iter 50 --max_samples 500
```

### Issue: Training unstable with OneCycleLR

**Solution**: Reduce max_lr or increase div_factor

```bash
--lr 0.05 --onecycle_div_factor 50.0
```

## Next Steps

1. **Run LR Finder**: `uv run python run_lr_finder.py --dataset imagenet1k`
2. **Examine Plot**: Check `./outputs/lr_finder_imagenet1k.png`
3. **Start Training**: Use suggested LR with OneCycleLR
4. **Monitor Progress**: Watch for convergence
5. **Evaluate Results**: Check test accuracy

## Advanced Topics

### Custom OneCycleLR Configuration

```bash
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 256 \
    --lr 0.1 \
    --epochs 100 \
    --scheduler onecycle \
    --onecycle_pct_start 0.4 \
    --onecycle_div_factor 10.0 \
    --onecycle_final_div_factor 1000.0 \
    --onecycle_anneal_strategy linear \
    --onecycle_three_phase
```

### Combining All Features

```bash
uv run python main.py \
    --dataset imagenet1k \
    --use_pretrained \
    --batch_size 256 \
    --lr 0.05 \
    --epochs 50 \
    --scheduler onecycle \
    --onecycle_pct_start 0.3 \
    --amp \
    --max_grad_norm 1.0 \
    --save_best \
    --snapshot_freq 5 \
    --weight_decay 1e-4
```

## Performance Expectations

### With Default Settings (CPU)
- **LR Finder**: ~5-10 minutes (1000 samples)
- **Training**: ~2-3 hours per epoch (full dataset)

### With GPU + AMP
- **LR Finder**: ~1-2 minutes (1000 samples)
- **Training**: ~10-20 minutes per epoch (full dataset)

### Convergence
- **OneCycleLR**: 50 epochs typically sufficient
- **Cosine**: 100-200 epochs recommended
- **Step**: 100-150 epochs recommended

## Documentation

- **README.md**: Complete project documentation
- **LR_FINDER_USAGE.md**: Detailed LR finder guide
- **ONECYCLE_LR_GUIDE.md**: OneCycleLR deep dive
- **IMPLEMENTATION_SUMMARY.md**: Technical implementation details

## Support

For issues or questions:
1. Check the documentation files
2. Run the test scripts
3. Review the example scripts
4. Open a GitHub issue

---

**Happy Training! 🚀**



