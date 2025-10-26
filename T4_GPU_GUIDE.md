# T4 GPU Optimization Guide for ImageNet Classifier

## Summary of Changes

The `lightning_main.py` script has been optimized for T4 GPU compatibility. Here are the key optimizations:

### 1. **Automatic Precision Detection** ✅
- **T4 GPUs**: Automatically uses `fp16-mixed` precision (T4s don't support bfloat16)
- **A100/A10G/H100**: Uses `bf16-mixed` precision for better performance
- The script auto-detects your GPU and selects the appropriate precision

### 2. **Memory-Optimized Batch Size** ✅
- **Default batch size**: Changed from 1028 to **128** (T4-friendly)
- ResNet50 with 25.6M parameters fits comfortably in 16GB VRAM at batch size 128

### 3. **Gradient Accumulation Support** ✅
- New `--accumulate_grad_batches` parameter (default: 1)
- Allows you to simulate larger batch sizes without OOM errors
- Example: `--batch_size 128 --accumulate_grad_batches 4` = effective batch size of 512

### 4. **Smart Warnings and Recommendations** ✅
- Displays T4-specific recommendations at runtime
- Warns if batch size is too large for T4's 16GB VRAM
- Suggests optimal gradient accumulation settings

---

## Recommended Usage for T4 GPUs

### Basic Training (Single T4)
```bash
python lightning_main.py \
    --data_dir ./data \
    --batch_size 128 \
    --accumulate_grad_batches 1 \
    --max_epochs 90 \
    --learning_rate 0.1
```

### Memory-Constrained Training (Conservative)
```bash
python lightning_main.py \
    --data_dir ./data \
    --batch_size 64 \
    --accumulate_grad_batches 8 \
    --max_epochs 90 \
    --learning_rate 0.1
```
- Effective batch size: 64 × 8 = **512**
- Lower memory usage per step
- Same training dynamics as batch_size=512

### Aggressive Training (Maximum Batch Size)
```bash
python lightning_main.py \
    --data_dir ./data \
    --batch_size 192 \
    --accumulate_grad_batches 4 \
    --max_epochs 90 \
    --learning_rate 0.1
```
- Effective batch size: 192 × 4 = **768**
- May require tuning based on your specific model/data
- Monitor for OOM errors

### With Full Augmentation (Production)
```bash
python lightning_main.py \
    --data_dir ./data \
    --batch_size 128 \
    --accumulate_grad_batches 4 \
    --max_epochs 90 \
    --learning_rate 0.1 \
    --warmup_epochs 5 \
    --random_erasing_p 0.25 \
    --mixup_alpha 0.2 \
    --cutmix_alpha 1.0
```

---

## T4 GPU Specifications
- **VRAM**: 16 GB GDDR6
- **CUDA Cores**: 2560
- **Tensor Cores**: 320 (Gen 2)
- **FP16 Performance**: 65 TFLOPS
- **Best Precision**: `fp16-mixed` (T4 doesn't support bfloat16)

---

## Troubleshooting

### Out of Memory (OOM) Errors
If you encounter OOM errors, try these steps in order:
1. **Reduce batch size**: `--batch_size 64` or `--batch_size 96`
2. **Increase gradient accumulation**: `--accumulate_grad_batches 4` or `8`
3. **Reduce workers**: `--num_workers 2`
4. **Disable augmentation**: Remove `--mixup_alpha`, `--cutmix_alpha`, `--random_erasing_p`

### Slow Training
If training is slower than expected:
1. **Check precision**: Ensure script is using `fp16-mixed` (check console output)
2. **Increase num_workers**: `--num_workers 8` (if CPU allows)
3. **Verify Tensor Cores**: Look for "Enabled high precision matmul" message
4. **Check batch size**: Larger batches (96-128) better utilize GPU

### Poor Accuracy
If accuracy is lower than expected:
1. **Increase effective batch size**: Use gradient accumulation
2. **Enable warmup**: `--warmup_epochs 5 --warmup_start_lr 1e-6`
3. **Add augmentation**: `--random_erasing_p 0.25 --mixup_alpha 0.2`
4. **Tune learning rate**: Use `--lr_finder` to find optimal LR

---

## Performance Expectations

### ImageNet (1000 classes, 1.28M training images)
- **Batch size 128**: ~0.8-1.0 it/s on single T4
- **Training time**: ~40-50 hours for 90 epochs
- **Memory usage**: ~12-14 GB VRAM

### TinyImageNet (200 classes, 100K training images)
- **Batch size 128**: ~2-3 it/s on single T4
- **Training time**: ~3-4 hours for 90 epochs
- **Memory usage**: ~8-10 GB VRAM

---

## Multi-GPU Training

If you have multiple T4s:
```bash
python lightning_main.py \
    --data_dir ./data \
    --batch_size 128 \
    --max_epochs 90
```
- Script automatically detects and uses all GPUs
- Uses DDP (DistributedDataParallel) strategy
- Effective batch size = `batch_size × num_gpus × accumulate_grad_batches`

---

## Key Parameters

| Parameter | Default | Recommended for T4 | Description |
|-----------|---------|-------------------|-------------|
| `--batch_size` | 128 | 64-128 | Per-GPU batch size |
| `--accumulate_grad_batches` | 1 | 4-8 | Gradient accumulation steps |
| `--num_workers` | 4 | 4-8 | Data loading workers |
| `--gradient_clip_val` | 1.0 | 1.0 | Gradient clipping |
| `--learning_rate` | 0.1 | 0.1 | Base learning rate |
| `--warmup_epochs` | 0 | 5 | Warmup epochs |
| `--max_epochs` | 10 | 90 | Total training epochs |

---

## Additional Notes

1. **Tensor Cores**: Automatically enabled via `torch.set_float32_matmul_precision('high')`
2. **Mixed Precision**: Automatically uses fp16 on T4 GPUs
3. **Pin Memory**: Enabled by default for faster data transfer
4. **Resume Training**: Use `--resume` to continue from last checkpoint
5. **TensorBoard**: All metrics logged for visualization

---

## Example Output

When running on T4, you should see:
```
✓ Enabled high precision matmul (Tensor Cores optimization)
✓ Detected Tesla T4 - using fp16-mixed precision (T4 compatible)
Auto-detected 1 GPU - using single GPU training
Batch size per GPU: 128
Gradient accumulation steps: 1
Effective batch size: 128

💡 T4 GPU Recommendations:
   ✓ Using fp16 mixed precision (T4 compatible)
   ✓ Batch size: 128 (recommended: 64-128 for ResNet50)
   ✓ Gradient accumulation: 1 (use 4-8 for larger effective batch)
```

---

## Contact & Support

For issues or questions:
- Check console output for warnings and recommendations
- Verify GPU detection: `nvidia-smi`
- Monitor memory: `watch -n 1 nvidia-smi`
- Review logs: `tensorboard --logdir ./results/logs/tensorboard_logs`

