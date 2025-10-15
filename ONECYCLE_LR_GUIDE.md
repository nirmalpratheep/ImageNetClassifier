# OneCycleLR Scheduler Guide

A comprehensive guide to using PyTorch's OneCycleLR scheduler for super-convergence training.

## Overview

OneCycleLR implements the 1cycle learning rate policy described in the paper ["Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"](https://arxiv.org/abs/1708.07120) by Leslie N. Smith and Nicholay Topin.

**Key Benefits:**
- Faster convergence (10× faster in some cases)
- Better generalization
- Larger batch sizes possible
- Fewer epochs needed

**Reference**: [PyTorch OneCycleLR Documentation](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)

## How It Works

### Two-Phase Policy (Default - fastai style)

1. **Phase 1 - Warmup (30% of training)**:
   - Learning rate increases from `initial_lr` to `max_lr`
   - Momentum decreases from `max_momentum` (0.95) to `base_momentum` (0.85)

2. **Phase 2 - Annealing (70% of training)**:
   - Learning rate decreases from `max_lr` to `min_lr`
   - Momentum increases from `base_momentum` (0.85) to `max_momentum` (0.95)

### Three-Phase Policy (Original paper)

Enable with `--onecycle_three_phase`:

1. **Phase 1 - Warmup**: LR increases to max_lr
2. **Phase 2 - Annealing**: LR decreases symmetrically
3. **Phase 3 - Fine-tuning**: LR decreases to very low value

## Learning Rate Calculation

```
initial_lr = max_lr / div_factor
min_lr = initial_lr / final_div_factor
```

**Example with default values:**
```
max_lr = 0.1 (your --lr argument)
div_factor = 25.0
final_div_factor = 10000.0

initial_lr = 0.1 / 25.0 = 0.004
min_lr = 0.004 / 10000.0 = 0.0000004
```

## Usage Examples

### Basic Usage

```bash
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 128 \
    --lr 0.1 \
    --epochs 50 \
    --scheduler onecycle
```

### With LR Finder

```bash
# Step 1: Find optimal max_lr
uv run python run_lr_finder.py --dataset imagenet1k --max_samples 1000

# Output: Suggested learning rate: 8.35e-02

# Step 2: Train with OneCycleLR
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 128 \
    --lr 0.0835 \
    --epochs 50 \
    --scheduler onecycle
```

### Custom Parameters

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
    --amp
```

### Three-Phase (Original Paper)

```bash
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 128 \
    --lr 0.1 \
    --epochs 90 \
    --scheduler onecycle \
    --onecycle_three_phase \
    --onecycle_pct_start 0.45
```

## Parameters Explained

### `--lr` (max_lr)
- **The peak learning rate** during training
- Use LR finder to discover optimal value
- Typically much higher than traditional training (e.g., 0.1 vs 0.01)

### `--onecycle_pct_start`
- **Percentage of cycle spent in warmup**
- Default: 0.3 (30% warmup, 70% annealing)
- Paper recommendation: 0.3-0.45
- Lower values = more time at high LR
- Higher values = gentler warmup

### `--onecycle_div_factor`
- **Controls initial learning rate**
- `initial_lr = max_lr / div_factor`
- Default: 25.0
- Smaller values = start with higher LR
- Larger values = gentler start

### `--onecycle_final_div_factor`
- **Controls minimum learning rate**
- `min_lr = initial_lr / final_div_factor`
- Default: 10000.0
- Larger values = lower final LR (more fine-tuning)

### `--onecycle_anneal_strategy`
- **How LR decreases in Phase 2**
- Options: `cos` (cosine) or `linear`
- Default: `cos`
- Cosine is generally smoother and more stable

### `--onecycle_three_phase`
- **Enable three-phase schedule**
- Default: False (two-phase, fastai style)
- True: Follows original paper more closely

## Momentum Cycling

OneCycleLR also cycles momentum **inversely** to learning rate:

- When LR is **low** → Momentum is **high** (0.95)
- When LR is **high** → Momentum is **low** (0.85)

This helps with:
- Faster convergence at high LR
- Better fine-tuning at low LR

**Note**: Momentum cycling only works with SGD optimizer.

## Best Practices

### 1. Finding Max LR

**Always use LR finder first:**
```bash
uv run python run_lr_finder.py --dataset imagenet1k --max_samples 2000
```

The suggested LR from the finder is your `max_lr` for OneCycleLR.

### 2. Batch Size

OneCycleLR works well with **larger batch sizes**:
- Start with 128-256
- Can go up to 512-1024 with good max_lr
- Larger batches = faster training

### 3. Number of Epochs

OneCycleLR achieves convergence **faster**:
- Reduce epochs by 2-5×
- 20-50 epochs often sufficient for ImageNet
- Monitor validation accuracy for early stopping

### 4. Warmup Percentage

**Recommendations:**
- 0.3: Good default (fastai)
- 0.4-0.45: Original paper recommendation
- 0.2: More aggressive, less stable
- 0.5: Very gentle, slower convergence

### 5. Div Factors

**Conservative (safer):**
```bash
--onecycle_div_factor 25.0
--onecycle_final_div_factor 10000.0
```

**Aggressive (faster, less stable):**
```bash
--onecycle_div_factor 10.0
--onecycle_final_div_factor 1000.0
```

## Comparison with Other Schedulers

| Feature | OneCycleLR | Cosine | StepLR |
|---------|------------|--------|--------|
| Convergence Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Stability | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Final Accuracy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Easy to Configure | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Needs LR Finder | Yes | Optional | Optional |
| Batch Size Flexibility | High | Medium | Medium |

## Common Issues & Solutions

### Issue 1: Training Unstable

**Solution**: Reduce max_lr or increase div_factor
```bash
--lr 0.05 --onecycle_div_factor 50.0
```

### Issue 2: Poor Final Accuracy

**Solution**: Increase final_div_factor for more fine-tuning
```bash
--onecycle_final_div_factor 50000.0
```

### Issue 3: Convergence Too Slow

**Solution**: Increase max_lr or decrease pct_start
```bash
--lr 0.15 --onecycle_pct_start 0.2
```

### Issue 4: Loss Diverges Early

**Solution**: Use LR finder to find appropriate max_lr
```bash
# First run LR finder, then use discovered LR
uv run python run_lr_finder.py --dataset imagenet1k
```

## Advanced Tips

### Tip 1: Combine with Mixed Precision

OneCycleLR works excellently with AMP:
```bash
uv run python main.py \
    --scheduler onecycle \
    --lr 0.1 \
    --amp
```

Benefits:
- 2-3× faster training
- Can use larger batch sizes
- Same or better accuracy

### Tip 2: Adjust for Small Datasets

For smaller datasets (like CIFAR-100):
```bash
--lr 0.05 \
--onecycle_pct_start 0.4 \
--onecycle_div_factor 10.0
```

### Tip 3: Fine-tuning Pretrained Models

Use **lower max_lr** for fine-tuning:
```bash
uv run python main.py \
    --dataset imagenet1k \
    --use_pretrained \
    --scheduler onecycle \
    --lr 0.01 \
    --onecycle_div_factor 10.0 \
    --onecycle_final_div_factor 1000.0
```

### Tip 4: Very Long Training

For 200+ epochs, use three-phase:
```bash
--scheduler onecycle \
--onecycle_three_phase \
--onecycle_pct_start 0.45 \
--epochs 200
```

## Scheduler Output Example

When you run OneCycleLR, you'll see:

```
[OneCycleLR] PyTorch Official Implementation:
   - Max LR: 0.1000
   - Epochs: 50
   - Steps per epoch: 1563
   - Total steps: 78150
   - Initial LR: 0.004000 (max_lr / 25.0)
   - Min LR: 0.00000040 (initial_lr / 10000.0)
   - Anneal strategy: cos
   - Percent start: 30% (warmup phase)
   - Three phase: False
   - Cycle momentum: True (0.85 <-> 0.95)
```

## Troubleshooting

### Q: Why is my LR not changing?

**A**: OneCycleLR steps **per batch**, not per epoch. Make sure:
- You're not manually stepping the scheduler
- The scheduler is passed to `train_epoch()`

### Q: What's the best pct_start value?

**A**: 
- 0.3: fastai recommendation (good default)
- 0.3-0.45: Generally works well
- Use LR finder plot to guide: more unstable region = higher pct_start

### Q: Should I use two-phase or three-phase?

**A**:
- **Two-phase (default)**: Faster convergence, good for most cases
- **Three-phase**: Original paper, better for very long training

### Q: How do I know if it's working?

**A**: You should see:
- Fast initial accuracy gains
- Smooth training curves
- Higher final accuracy in fewer epochs
- LR changing every batch in logs

## Example Training Run

```bash
# Complete example: LR finder + OneCycleLR training
uv run python run_lr_finder.py \
    --dataset imagenet1k \
    --batch_size 64 \
    --max_samples 2000

# Use suggested LR (e.g., 0.085) for training
uv run python main.py \
    --dataset imagenet1k \
    --batch_size 256 \
    --lr 0.085 \
    --epochs 50 \
    --scheduler onecycle \
    --onecycle_pct_start 0.3 \
    --onecycle_anneal_strategy cos \
    --amp \
    --max_grad_norm 1.0 \
    --save_best \
    --snapshot_dir ./models
```

Expected results:
- **Epoch 15** (30% complete): LR reaches max (0.085), accuracy ~40%
- **Epoch 30**: LR decreasing, accuracy ~65%
- **Epoch 50**: LR at minimum, accuracy ~75-80%

## References

- [PyTorch OneCycleLR Documentation](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
- [Super-Convergence Paper](https://arxiv.org/abs/1708.07120)
- [fastai 1cycle implementation](https://docs.fast.ai/callback.schedule.html#OneCycleScheduler)
- [Leslie Smith's blog posts on LR policies](https://sgugger.github.io/the-1cycle-policy.html)

## Summary

OneCycleLR is a powerful scheduler that:
- ✅ Trains faster (fewer epochs needed)
- ✅ Achieves better accuracy
- ✅ Works with large batch sizes
- ✅ Automatically handles warmup and annealing
- ⚠️ Requires LR finder to set max_lr
- ⚠️ Steps per batch (different from other schedulers)

For best results: **Use LR Finder → Set max_lr → Use OneCycleLR**



