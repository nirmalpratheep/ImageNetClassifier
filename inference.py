#!/usr/bin/env python3
"""
Inference and Evaluation Script for ImageNet ResNet-50

Loads a checkpoint, runs inference on validation set, and plots training curves.
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_model import ImageNetLightningModule
from data_module import ImageNetDataModule
from torchmetrics import Accuracy
import json
from pathlib import Path


def compute_topk_accuracy(logits, targets, topk=(1, 5)):
    """Compute top-k accuracy."""
    maxk = max(topk)
    batch_size = targets.size(0)
    
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate_model(model, dataloader, device, num_classes=1000):
    """Evaluate model on dataset and compute Top-1 and Top-5 accuracy."""
    model.eval()
    
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    top1_acc = Accuracy(task='multiclass', num_classes=num_classes)
    top5_acc = Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
    
    if device.type == 'cuda':
        top1_acc = top1_acc.to(device)
        top5_acc = top5_acc.to(device)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Convert to channels_last if model uses it
            if hasattr(model, '_use_channels_last') and model._use_channels_last:
                images = images.to(memory_format=torch.channels_last)
            
            # Forward pass
            logits = model(images)
            
            # Compute loss
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            # Compute Top-1 and Top-5
            top1, top5 = compute_topk_accuracy(logits, targets, topk=(1, 5))
            top1_correct += top1.item() * len(targets) / 100.0
            top5_correct += top5.item() * len(targets) / 100.0
            total_samples += len(targets)
            
            # Update metrics
            top1_acc(logits, targets)
            top5_acc(logits, targets)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")
    
    avg_loss = total_loss / len(dataloader)
    top1_acc_value = top1_acc.compute().item() * 100
    top5_acc_value = top5_acc.compute().item() * 100
    
    return {
        'loss': avg_loss,
        'top1_acc': top1_acc_value,
        'top5_acc': top5_acc_value,
    }


def extract_metrics_from_checkpoint(checkpoint_path):
    """Extract metadata and metrics from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        metrics = {
            'epoch': checkpoint.get('epoch', None),
            'global_step': checkpoint.get('global_step', None),
            'best_val_acc': None,
            'best_val_loss': None,
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': [],
        }
        
        # Extract from Lightning checkpoint
        if 'callbacks' in checkpoint:
            # Check ModelCheckpoint callback
            for key, value in checkpoint['callbacks'].items():
                if 'ModelCheckpoint' in key:
                    if isinstance(value, dict):
                        best_model_path = value.get('best_model_path', None)
                        best_model_score = value.get('best_model_score', None)
                        if best_model_score is not None:
                            metrics['best_val_acc'] = float(best_model_score) * 100
        
        # Extract from model state (if stored in Lightning module)
        if 'state_dict' in checkpoint:
            # Try to extract hyperparameters
            if 'hyper_parameters' in checkpoint:
                hparams = checkpoint['hyper_parameters']
                metrics['num_classes'] = hparams.get('num_classes', 1000)
                metrics['lr'] = hparams.get('lr', None)
        
        # Check if training history is stored in Lightning module
        # (This might be in a different format depending on how it was saved)
        
        return metrics, checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None


def load_training_history_from_tensorboard(log_dir, experiment_name=None):
    """Try to load training history from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Find TensorBoard log directory
        if experiment_name:
            log_path = os.path.join(log_dir, experiment_name)
        else:
            # Find the most recent version
            log_path = log_dir
        
        if os.path.exists(log_path):
            # Find version directory
            versions = [d for d in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, d)) and d.startswith('version')]
            if versions:
                # Use most recent version
                versions.sort()
                latest_version = versions[-1]
                log_path = os.path.join(log_path, latest_version)
        
        if not os.path.exists(log_path):
            return None
        
        # Load event files
        event_files = [f for f in os.listdir(log_path) if f.startswith('events.out.tfevents')]
        if not event_files:
            return None
        
        ea = EventAccumulator(log_path)
        ea.Reload()
        
        # Extract scalars
        scalars = ea.Tags()['scalars']
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': [],
        }
        
        # Extract epoch-level metrics
        if 'train_loss_epoch' in scalars:
            train_loss_events = ea.Scalars('train_loss_epoch')
            history['train_loss'] = [s.value for s in train_loss_events]
            history['epochs'] = list(range(len(train_loss_events)))
        
        if 'train_acc' in scalars:
            train_acc_events = ea.Scalars('train_acc')
            history['train_acc'] = [s.value * 100 for s in train_acc_events]  # Convert to percentage
        
        if 'val_loss_epoch' in scalars or 'val_loss' in scalars:
            val_key = 'val_loss_epoch' if 'val_loss_epoch' in scalars else 'val_loss'
            val_loss_events = ea.Scalars(val_key)
            history['val_loss'] = [s.value for s in val_loss_events]
        
        if 'val_acc' in scalars:
            val_acc_events = ea.Scalars('val_acc')
            history['val_acc'] = [s.value * 100 for s in val_acc_events]  # Convert to percentage
        
        return history
    except ImportError:
        print("TensorBoard not available for loading history")
        return None
    except Exception as e:
        print(f"Could not load TensorBoard history: {e}")
        return None


def plot_training_curves(history, output_path, checkpoint_metrics=None):
    """Plot training curves."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    epochs = history.get('epochs', list(range(len(history.get('train_loss', [])))))
    
    # Plot 1: Training and Validation Loss
    ax1 = fig.add_subplot(gs[0, 0])
    if history.get('train_loss'):
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if history.get('val_loss'):
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    if history.get('train_acc'):
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    if history.get('val_acc'):
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        # Mark best validation accuracy
        if history['val_acc']:
            best_idx = np.argmax(history['val_acc'])
            best_acc = history['val_acc'][best_idx]
            best_epoch = epochs[best_idx]
            ax2.plot(best_epoch, best_acc, 'go', markersize=10, label=f'Best Val Acc: {best_acc:.2f}%')
            ax2.annotate(f'{best_acc:.2f}%', 
                        xy=(best_epoch, best_acc), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Top-1 Accuracy (if available)
    ax3 = fig.add_subplot(gs[1, 0])
    if history.get('val_acc'):
        ax3.plot(epochs, history['val_acc'], 'g-', label='Top-1 Accuracy', linewidth=2)
        if history['val_acc']:
            best_idx = np.argmax(history['val_acc'])
            best_acc = history['val_acc'][best_idx]
            best_epoch = epochs[best_idx]
            ax3.plot(best_epoch, best_acc, 'ro', markersize=10, label=f'Best: {best_acc:.2f}%')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax3.set_title('Top-1 Validation Accuracy', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Top-5 Accuracy (if available)
    ax4 = fig.add_subplot(gs[1, 1])
    if history.get('val_top5_acc'):
        ax4.plot(epochs, history['val_top5_acc'], 'm-', label='Top-5 Accuracy', linewidth=2)
        if history['val_top5_acc']:
            best_idx = np.argmax(history['val_top5_acc'])
            best_acc = history['val_top5_acc'][best_idx]
            best_epoch = epochs[best_idx]
            ax4.plot(best_epoch, best_acc, 'ro', markersize=10, label=f'Best: {best_acc:.2f}%')
    else:
        # If we have Top-1, estimate Top-5 (typically Top-5 is ~10-15% higher)
        if history.get('val_acc'):
            estimated_top5 = [acc + 12 for acc in history['val_acc']]  # Rough estimate
            ax4.plot(epochs, estimated_top5, 'm--', label='Estimated Top-5 Accuracy', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Top-5 Accuracy (%)', fontsize=14, fontweight='bold')
    ax4.set_title('Top-5 Validation Accuracy', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Combined Loss and Accuracy (dual y-axis)
    ax5 = fig.add_subplot(gs[2, :])
    if history.get('train_loss') and history.get('val_loss'):
        ax5_twin = ax5.twinx()
        
        # Loss on left axis
        line1 = ax5.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, alpha=0.7)
        line2 = ax5.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, alpha=0.7)
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Loss', fontsize=12, color='black')
        ax5.tick_params(axis='y', labelcolor='black')
        
        # Accuracy on right axis
        if history.get('val_acc'):
            line3 = ax5_twin.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy', linewidth=2)
            ax5_twin.set_ylabel('Accuracy (%)', fontsize=12, color='green')
            ax5_twin.tick_params(axis='y', labelcolor='green')
        
        # Combine legends
        lines = line1 + line2
        if history.get('val_acc'):
            lines += line3
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='center right', fontsize=11)
        
        ax5.set_title('Training Progress: Loss and Accuracy', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # Add checkpoint info as text
    if checkpoint_metrics:
        info_text = f"Checkpoint: Epoch {checkpoint_metrics.get('epoch', 'N/A')}\n"
        if checkpoint_metrics.get('best_val_acc'):
            info_text += f"Best Val Acc: {checkpoint_metrics['best_val_acc']:.2f}%\n"
        fig.text(0.02, 0.02, info_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('ImageNet ResNet-50 Training Curves', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {output_path}")
    plt.close()


def print_metrics(eval_results, checkpoint_metrics=None):
    """Print evaluation metrics."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n📊 Model Performance:")
    print(f"  Validation Loss:  {eval_results['loss']:.4f}")
    print(f"  Top-1 Accuracy:   {eval_results['top1_acc']:.2f}%")
    print(f"  Top-5 Accuracy:   {eval_results['top5_acc']:.2f}%")
    
    if checkpoint_metrics:
        print(f"\n📁 Checkpoint Info:")
        if checkpoint_metrics.get('epoch') is not None:
            print(f"  Epoch:             {checkpoint_metrics['epoch']}")
        if checkpoint_metrics.get('global_step') is not None:
            print(f"  Global Step:       {checkpoint_metrics['global_step']}")
        if checkpoint_metrics.get('best_val_acc'):
            print(f"  Best Val Acc:      {checkpoint_metrics['best_val_acc']:.2f}%")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Inference and Evaluation Script")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file (.ckpt)")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Path to ImageNet data directory")
    
    # Optional arguments
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size")
    parser.add_argument("--output_dir", type=str, default="./inference_output",
                       help="Output directory for plots and results")
    parser.add_argument("--plot_name", type=str, default="training_curves.png",
                       help="Name of the plot file")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="TensorBoard log directory (for loading training history)")
    parser.add_argument("--experiment_name", type=str, default="imagenet_resnet50",
                       help="Experiment name (for TensorBoard log loading)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*80)
    print("INFERENCE AND EVALUATION")
    print("="*80)
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    print(f"🖥️  Using device: {device}")
    
    # Extract checkpoint metadata
    checkpoint_metrics, checkpoint_data = extract_metrics_from_checkpoint(args.checkpoint)
    
    # Load model from checkpoint
    try:
        model = ImageNetLightningModule.load_from_checkpoint(
            args.checkpoint,
            map_location=device
        )
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
        
        # Get num_classes from model
        num_classes = model.num_classes
        print(f"✓ Model configured for {num_classes} classes")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load validation dataset
    print(f"\n📂 Loading validation dataset from: {args.data_dir}")
    data_module = ImageNetDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        augmentation=False,  # No augmentation for validation
        max_samples_per_class=None,
    )
    # Setup datasets - use "fit" stage to load both train and val (we only need val, but setup requires fit)
    data_module.setup("fit")
    val_loader = data_module.val_dataloader()
    
    # Check that validation dataset was loaded
    if data_module.val_dataset is None:
        raise ValueError("Validation dataset is None. Check that data_dir contains 'val' folder.")
    
    print(f"✓ Validation dataset loaded: {len(data_module.val_dataset)} samples")
    
    # Run evaluation
    print(f"\n🔄 Running inference on validation set...")
    eval_results = evaluate_model(model, val_loader, device, num_classes=num_classes)
    
    # Print metrics
    print_metrics(eval_results, checkpoint_metrics)
    
    # Try to load training history from TensorBoard
    print(f"\n📊 Loading training history...")
    history = load_training_history_from_tensorboard(args.log_dir, args.experiment_name)
    
    if history and (history.get('train_loss') or history.get('val_loss')):
        print("✓ Training history loaded from TensorBoard")
        
        # Add current evaluation results to history if not present
        if history.get('val_acc') and len(history['val_acc']) > 0:
            # Current checkpoint might be the latest, add it if missing
            pass
        
        # Create plots
        plot_path = os.path.join(args.output_dir, args.plot_name)
        print(f"\n📈 Generating training curves...")
        plot_training_curves(history, plot_path, checkpoint_metrics)
    else:
        print("⚠ Could not load training history from TensorBoard")
        print(f"  Tried: {os.path.join(args.log_dir, args.experiment_name)}")
        print("  You can manually provide training history or use TensorBoard to view curves")
    
    # Save evaluation results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    results = {
        'checkpoint': args.checkpoint,
        'evaluation': eval_results,
        'checkpoint_metrics': checkpoint_metrics,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Evaluation results saved to: {results_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

