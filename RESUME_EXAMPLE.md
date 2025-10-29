# How to Resume Training from Checkpoint

## Quick Example

After training completes, you'll see output like:
```
Best checkpoint: ./checkpoints/epoch=00-val_acc=0.0008-val_loss=7.1297.ckpt
Latest checkpoint: ./checkpoints/last.ckpt
```

### Option 1: Resume from Latest Checkpoint (Recommended)

The `last.ckpt` is saved after every epoch and is the easiest to resume from:

```bash
python main_lightning.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 50 \
    --resume_from ./checkpoints/last.ckpt
```

### Option 2: Resume from Best Checkpoint

Resume from the checkpoint with the best validation accuracy:

```bash
python main_lightning.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 50 \
    --resume_from ./checkpoints/epoch=00-val_acc=0.0008-val_loss=7.1297.ckpt
```

### Option 3: Resume from Specific Epoch

If you want to resume from a specific epoch checkpoint:

```bash
python main_lightning.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 100 \
    --resume_from ./checkpoints/epoch=25-val_acc=0.5432-val_loss=2.3456.ckpt
```

## What Gets Restored

When you resume from a checkpoint:

✅ **Model weights** - All trained parameters
✅ **Optimizer state** - Momentum, etc. are preserved
✅ **Scheduler state** - Learning rate schedule continues correctly
✅ **Epoch number** - Training continues from the next epoch
✅ **Hyperparameters** - All training config is preserved
✅ **Metrics history** - Training/validation metrics continue

## Important Notes

1. **Continue Training for More Epochs**
   ```bash
   # Original training: 50 epochs, stopped at epoch 25
   # Resume for another 50 epochs (will train to epoch 75)
   python main_lightning.py \
       --resume_from ./checkpoints/epoch=25-val_acc=0.5432.ckpt \
       --epochs 100  # Train until epoch 100 total
   ```

2. **All Arguments Must Match**
   - Keep the same model architecture (num_classes, etc.)
   - Keep the same data directory structure
   - Can change: `--epochs`, `--batch_size` (if supported)

3. **Checkpoint Contains Everything**
   - The checkpoint file has everything needed to resume
   - No need to specify learning rate, scheduler, etc. again
   - They're all saved in the checkpoint

4. **TensorBoard Logs Continue**
   - TensorBoard logs will append to the same run
   - Or create a new version by specifying `--version`

## Example: Continuing Training

```bash
# Initial training
python main_lightning.py --epochs 25 --name my_experiment

# Resume for more epochs
python main_lightning.py \
    --resume_from ./checkpoints/last.ckpt \
    --epochs 50 \
    --name my_experiment  # Same name to continue logs
```

## Troubleshooting

**"Checkpoint not found"**
- Make sure the path is correct (use absolute path if needed)
- Check that the checkpoint file exists

**"Mismatched hyperparameters"**
- The checkpoint saves all hyperparameters
- Lightning will warn if there are mismatches
- Usually safe to ignore if you're sure

**"CUDA out of memory after resume"**
- Reduce batch size: `--batch_size 128`
- Or the model state may have changed

