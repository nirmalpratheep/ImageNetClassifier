# ImageNet Subset Data Structure Guide

This guide explains how to use the ImageNet-1K classifier with your ImageNet subset data structure.

## Your Data Structure

Your data is organized as follows:
```
./data/
├── 00500/
│   ├── filename1.jpg
│   ├── filename2.jpg
│   └── ...
├── 00501/
│   ├── filename1.jpg
│   ├── filename2.jpg
│   └── ...
...
├── 00993/
│   ├── filename1.jpg
│   ├── filename2.jpg
│   └── ...
```

This is a **ImageNet subset** format where:
- Each folder represents a class (00500, 00501, etc.)
- The folder names are ImageNet class IDs
- Images are directly in the class folders

## How It Works

The system automatically:
1. **Detects** your data structure as "imagenet_subset"
2. **Scans** all class directories and counts images
3. **Creates** train/validation splits (default 80%/20%)
4. **Applies** appropriate transforms for training and validation
5. **Maps** numeric class IDs to sequential labels (0, 1, 2, ...)

## Quick Start

### 1. Test Your Data Structure

First, validate that your data works correctly:

```bash
# Test your data structure
uv run python test_data_structure.py --data_dir ./data

# Test with limited samples
uv run python test_data_structure.py --data_dir ./data --max_samples 500
```

This will show:
- ✅ Data structure detection
- ✅ Number of classes and samples
- ✅ Train/val split information
- ✅ Sample batch loading

### 2. Run LR Finder

Use the safe LR finder with good defaults:

```bash
# Basic LR finder (recommended for first try)
uv run python run_lr_finder_safe.py \
    --data_dir ./data \
    --max_samples 2000 \
    --batch_size 128

# Full dataset LR finder
uv run python run_lr_finder_safe.py \
    --data_dir ./data \
    --batch_size 256
```

### 3. Train Your Model

After finding the optimal learning rate:

```bash
# Train with found LR
uv run python train_with_lr.py \
    --data_dir ./data \
    --auto_lr \
    --batch_size 256 \
    --max_samples 10000  # Remove for full dataset
```

## Advanced Usage

### Custom Validation Split

You can adjust the train/validation split ratio:

```bash
# 90% train, 10% validation
uv run python run_lr_finder_safe.py \
    --data_dir ./data \
    --val_ratio 0.1

# 70% train, 30% validation  
uv run python run_lr_finder_safe.py \
    --data_dir ./data \
    --val_ratio 0.3
```

### With Wandb Logging

Track your experiments with Weights & Biases:

```bash
# LR finder with wandb
uv run python run_lr_finder_safe.py \
    --data_dir ./data \
    --use_wandb \
    --wandb_project "imagenet-subset-experiments" \
    --wandb_tags subset lr_finder

# Training with wandb
uv run python train_with_lr.py \
    --data_dir ./data \
    --auto_lr \
    --use_wandb \
    --wandb_project "imagenet-subset-training"
```

## Configuration Options

### Data Loading Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `./data` | Path to your data directory |
| `--val_ratio` | `0.2` | Validation split ratio (0.1 = 10%, 0.2 = 20%, etc.) |
| `--max_samples` | `None` | Limit samples for testing (e.g., 1000, 5000) |

### LR Finder Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr_start` | `1e-5` | Starting learning rate |
| `--lr_end` | `1.0` | Ending learning rate |
| `--lr_iter` | `200` | Number of iterations |
| `--batch_size` | `256` | Batch size |

## Example Workflows

### 1. Quick Test (Small Dataset)

```bash
# Step 1: Test data structure
uv run python test_data_structure.py --data_dir ./data --max_samples 1000

# Step 2: Quick LR finder
uv run python run_lr_finder_safe.py \
    --data_dir ./data \
    --max_samples 1000 \
    --batch_size 32 \
    --lr_iter 50

# Step 3: Quick training test
uv run python main.py \
    --data_dir ./data \
    --max_samples 1000 \
    --lr 0.001 \
    --epochs 2 \
    --batch_size 32
```

### 2. Full Dataset Training

```bash
# Step 1: Validate full dataset
uv run python test_data_structure.py --data_dir ./data

# Step 2: Find optimal LR
uv run python run_lr_finder_safe.py \
    --data_dir ./data \
    --batch_size 256 \
    --lr_iter 500 \
    --use_wandb

# Step 3: Full training
uv run python train_with_lr.py \
    --data_dir ./data \
    --auto_lr \
    --batch_size 256 \
    --use_wandb
```

### 3. Experiment with Different Splits

```bash
# Test different validation ratios
for ratio in 0.1 0.2 0.3; do
    uv run python run_lr_finder_safe.py \
        --data_dir ./data \
        --val_ratio $ratio \
        --max_samples 2000 \
        --wandb_run_name "lr_finder_val_${ratio}" \
        --use_wandb
done
```

## Expected Output

### Data Structure Detection
```
Detected data structure: imagenet_subset
Loading ImageNet subset from: ./data
Found 494 classes with 122394 total samples
Classes: ['00500', '00501', '00502', '00503', '00504']...
Dataset split: 97915 train, 24479 validation
✓ ImageNet subset loaded successfully
✓ Classes: 494
✓ Train samples: 97915
✓ Val samples: 24479
```

### LR Finder Results
```
Learning rate search finished. See the graph with {finder_name}.plot()
Suggested learning rate: 3.45e-03
LR finder plot saved to: ./outputs/lr_finder_safe_imagenet.png
```

## Troubleshooting

### Common Issues

1. **"No data directory found"**
   ```bash
   # Check your data directory exists
   ls -la ./data/
   # Should show folders like 00500, 00501, etc.
   ```

2. **"Unknown data structure"**
   - Make sure your folders are numeric (00500, not class_500)
   - Ensure you have at least 5 class directories
   - Check that folders contain image files (.jpg, .png, etc.)

3. **"Very few samples detected"**
   - Verify your images have correct extensions
   - Check file permissions
   - Ensure images aren't corrupted

4. **Memory errors**
   ```bash
   # Reduce batch size and samples
   --batch_size 64 --max_samples 1000
   ```

### Validation Commands

```bash
# Check data structure
find ./data -name "*.jpg" | head -10

# Count classes and images
echo "Classes: $(ls ./data | wc -l)"
echo "Total images: $(find ./data -name "*.jpg" | wc -l)"

# Check class distribution
for dir in ./data/*/; do 
    echo "$(basename $dir): $(ls $dir/*.jpg 2>/dev/null | wc -l) images"
done | head -5
```

## Integration with HuggingFace

The system still supports HuggingFace datasets as a fallback:

1. **Priority**: Local data (your format) > HuggingFace > Dummy data
2. **Compatibility**: Your format works with all existing scripts
3. **Flexibility**: Mix and match different data sources

## Class Mapping

Your numeric class directories (00500, 00501, etc.) are automatically mapped to sequential indices:

```python
# Your classes: ['00500', '00501', '00502', ..., '00993'] 
# Mapped to:    [0,       1,       2,       ..., 493]
```

The model will output predictions for indices 0-493, which correspond to your ImageNet class IDs.

## Performance Tips

1. **For Development**: Use `--max_samples 1000-5000`
2. **For LR Finding**: Use full dataset or `--max_samples 10000+`
3. **For Training**: Use full dataset without `--max_samples`
4. **Memory**: Start with smaller batch sizes (64, 128) then increase
5. **Speed**: Use SSD storage and multiple workers (`--num_workers 8`)

This setup gives you full compatibility with the existing ImageNet training pipeline while handling your specific data structure automatically! 🚀
