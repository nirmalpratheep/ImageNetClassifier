import argparse
import importlib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime

from preprocess import get_data_loaders
from train import train_epoch, evaluate
from torch.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
from torchsummary import summary
from lr_finder import find_lr, find_lr_advanced, LRFinder
# Visualization imports - create dummy classes if module doesn't exist
try:
    from visualization import (
        create_training_summary, create_evaluation_summary, 
        evaluate_model_comprehensive, TrainingVisualizer, MetricsCalculator,
        CIFAR10_CLASSES, CIFAR100_CLASSES
    )
except ImportError:
    print("Warning: visualization module not found. Creating dummy implementations.")
    
    # Dummy classes for compatibility
    class MetricsCalculator:
        def __init__(self, num_classes):
            self.num_classes = num_classes
        def calculate_metrics(self, targets, predictions, probabilities):
            return {'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0}
    
    def create_training_summary(*args, **kwargs):
        pass
    
    def create_evaluation_summary(*args, **kwargs):
        pass
    
    def evaluate_model_comprehensive(*args, **kwargs):
        return {'targets': [], 'predictions': [], 'probabilities': []}
    
    CIFAR10_CLASSES = [f"class_{i}" for i in range(10)]
    CIFAR100_CLASSES = [f"class_{i}" for i in range(100)]




def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensures reproducibility for cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(model_name: str, device: torch.device, dataset_name: str = "cifar100", use_pretrained: bool = False):
    if dataset_name.lower() == "imagenet1k" or dataset_name.lower() == "imagenet":
        module = importlib.import_module("model_resnet50")
        num_classes = 1000  # ImageNet has 1000 classes
        input_size = 224
        
        if use_pretrained:
            return module.load_pretrained_resnet50(device, num_classes=num_classes, input_size=input_size)
        else:
            return module.build_model(device, num_classes=num_classes, input_size=input_size, model_type="resnet50")
    else:
        module = importlib.import_module("model_resnet50")
        num_classes = 100  # CIFAR-100 has 100 classes
        input_size = 32
        return module.build_model(device, num_classes=num_classes, input_size=input_size, model_type="resnet34")


def save_snapshot(model, optimizer, scheduler, epoch, train_losses, train_acc, test_losses, test_acc, 
                 snapshot_dir: str, model_name: str):
    """Save model snapshot with training state."""
    os.makedirs(snapshot_dir, exist_ok=True)
    
    snapshot = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'train_acc': train_acc,
        'test_losses': test_losses,
        'test_acc': test_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    snapshot_path = os.path.join(snapshot_dir, f"{model_name}_epoch_{epoch}.pth")
    torch.save(snapshot, snapshot_path)
    print(f"Snapshot saved: {snapshot_path}")
    return snapshot_path


def load_snapshot(snapshot_path: str, model, optimizer, scheduler, device):
    """Load model snapshot and return training state."""
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    
    print(f"Loading snapshot: {snapshot_path}")
    snapshot = torch.load(snapshot_path, map_location=device)
    
    model.load_state_dict(snapshot['model_state_dict'])
    optimizer.load_state_dict(snapshot['optimizer_state_dict'])
    scheduler.load_state_dict(snapshot['scheduler_state_dict'])
    
    epoch = snapshot['epoch']
    train_losses = snapshot.get('train_losses', [])
    train_acc = snapshot.get('train_acc', [])
    test_losses = snapshot.get('test_losses', [])
    test_acc = snapshot.get('test_acc', [])
    
    print(f"Resumed from epoch {epoch}")
    return epoch, train_losses, train_acc, test_losses, test_acc


def get_lr_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """Calculate learning rate warmup factor."""
    if epoch <= warmup_epochs:
        return epoch / warmup_epochs
    return 1.0


def apply_warmup_lr(optimizer, base_lr: float, warmup_factor: float):
    """Apply warmup learning rate to optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * warmup_factor


def main():
    parser = argparse.ArgumentParser(description="Image Classification Training")
    parser.add_argument("--model", type=str, default="resnet34", help="Model to use")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "imagenet1k"], help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--step_size", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs (disabled)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "onecycle"], help="Learning rate scheduler")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--no_cuda", action="store_true")
    
    # Dataset streaming arguments
    parser.add_argument("--streaming", action="store_true", default=True, help="Use streaming for large datasets (default: True)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use (for testing/debugging)")
    
    # Model arguments
    parser.add_argument("--use_pretrained", action="store_true", help="Use pretrained ResNet-50 weights from Microsoft")
    
    # LR Finder arguments
    parser.add_argument("--find_lr", action="store_true", help="Run learning rate finder")
    parser.add_argument("--lr_advanced", action="store_true", help="Use advanced LR finder with more options")
    parser.add_argument("--lr_start", type=float, default=1e-7, help="Starting learning rate for LR finder")
    parser.add_argument("--lr_end", type=float, default=10, help="Ending learning rate for LR finder")
    parser.add_argument("--lr_iter", type=int, default=100, help="Number of iterations for LR finder")
    parser.add_argument("--lr_plot", type=str, default="./lr_finder_plot.png", help="Path to save LR finder plot")
    parser.add_argument("--lr_step_mode", type=str, default="exp", choices=["exp", "linear"], help="LR step mode")
    parser.add_argument("--lr_smooth_f", type=float, default=0.05, help="Smoothing factor for LR finder")
    parser.add_argument("--lr_diverge_th", type=float, default=5, help="Divergence threshold for LR finder")
    
    # Snapshot-related arguments
    parser.add_argument("--snapshot_dir", type=str, default="./snapshots", help="Directory to save model snapshots")
    parser.add_argument("--snapshot_freq", type=int, default=5, help="Save snapshot every N epochs")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to snapshot file to resume training from")
    parser.add_argument("--save_best", action="store_true", help="Save snapshot only when test accuracy improves")
    
    # Visualization arguments
    parser.add_argument("--plot_dir", type=str, default="./plots", help="Directory to save plots and visualizations")
    parser.add_argument("--plot_training", action="store_true", help="Generate training curves and plots")
    parser.add_argument("--plot_evaluation", action="store_true", help="Generate evaluation plots (confusion matrix, metrics)")
    parser.add_argument("--plot_freq", type=int, default=10, help="Generate plots every N epochs")
    parser.add_argument("--no_plots", action="store_true", help="Disable all plotting")
    
    # Training features
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (AMP)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    args = parser.parse_args()
    set_seed(42)
    device = get_device(prefer_cuda=not args.no_cuda)

    # Optimize data loading based on device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    pin_memory = use_cuda  # Use pin_memory when CUDA is available for better performance
    num_workers = min(args.num_workers, 2) if not use_cuda else args.num_workers
    
    print(f"Data loading: num_workers={num_workers}, pin_memory={pin_memory}, use_cuda={use_cuda}")
    
    print(f"Loading {args.dataset} dataset...")
    if args.streaming:
        print("Using streaming mode - no full dataset download required")
    if args.max_samples:
        print(f"Limited to {args.max_samples} samples per split")
    
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle_train=True,
        dataset_name=args.dataset,
        streaming=args.streaming,
        max_samples=args.max_samples,
    )
    
    # Test data loading
    print("Testing data loading...")
    try:
        test_batch = next(iter(train_loader))
        print(f"Data loading successful! Batch shape: {test_batch[0].shape}, labels: {test_batch[1].shape}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return 1

    model = build_model(args.model, device, args.dataset, args.use_pretrained)
    print(f"Device: {device}")
    if args.use_pretrained:
        print("✓ Using pretrained ResNet-50 weights from Microsoft")
    print(f"Model loaded, starting training...")
    
    # Detect number of classes from model
    num_classes = model.fc.out_features if hasattr(model, 'fc') else 100
    if args.dataset.lower() == "imagenet1k":
        dataset_name = "ImageNet1K"
        input_size = 224
    else:
        dataset_name = "CIFAR-100"
        input_size = 32
    
    class_names = CIFAR100_CLASSES if num_classes == 100 else CIFAR10_CLASSES
    print(f"Dataset: {dataset_name} ({num_classes} classes)")
    
    # Show model summary
    summary(model, input_size=(3, input_size, input_size))
    
    # Improved optimizer with weight decay
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                         weight_decay=args.weight_decay, nesterov=True)
    
    # Run LR finder if requested
    if args.find_lr:
        print("\n" + "="*70)
        print("RUNNING LEARNING RATE FINDER")
        print("="*70)
        
        try:
            if args.lr_advanced:
                suggested_lr, fig = find_lr_advanced(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=nn.CrossEntropyLoss(),
                    device=device,
                    start_lr=args.lr_start,
                    end_lr=args.lr_end,
                    num_iter=args.lr_iter,
                    step_mode=args.lr_step_mode,
                    smooth_f=args.lr_smooth_f,
                    diverge_th=args.lr_diverge_th,
                    plot=True,
                    save_path=args.lr_plot
                )
            else:
                suggested_lr, fig = find_lr(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=nn.CrossEntropyLoss(),
                    device=device,
                    start_lr=args.lr_start,
                    end_lr=args.lr_end,
                    num_iter=args.lr_iter,
                    plot=True,
                    save_path=args.lr_plot,
                    use_amp=args.amp
                )
            
            print(f"\nSuggested learning rate: {suggested_lr:.2e}")
            print(f"LR finder plot saved to: {args.lr_plot}")
            
            # Update optimizer with suggested LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = suggested_lr
            
            print(f"Updated optimizer learning rate to: {suggested_lr:.2e}")
            
        except Exception as e:
            print(f"LR finder failed: {e}")
            print("Continuing with original learning rate...")
        
        print("="*70)
        print("LR FINDER COMPLETED")
        print("="*70 + "\n")
    
    # Improved scheduler setup
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "onecycle":
        # Custom One Cycle LR following the pattern:
        # Phase 1: 41 epochs from 0.08 -> 0.8
        # Phase 2: 41 epochs from 0.8 -> 0.08
        # Phase 3: 18 epochs from 0.08 -> 0.008 (annihilation)
        
        min_lr = 0.01
        max_lr = 0.1
        annihilation_lr = 0.001  # 1/10th of min_lr
        
        total_epochs = args.epochs
        up_epochs = 41
        down_epochs = 41
        annihilate_epochs = total_epochs - up_epochs - down_epochs  # 18 for 100 epochs
        
        print(f"\n[OneCycle LR] Custom Setup:")
        print(f"   - Total Epochs: {total_epochs}")
        print(f"   - Phase 1 (warmup): {up_epochs} epochs | LR: {min_lr:.3f} -> {max_lr:.3f}")
        print(f"   - Phase 2 (cooldown): {down_epochs} epochs | LR: {max_lr:.3f} -> {min_lr:.3f}")
        print(f"   - Phase 3 (annihilation): {annihilate_epochs} epochs | LR: {min_lr:.3f} -> {annihilation_lr:.3f}\n")
        
        def lr_lambda(epoch):
            """
            Custom OneCycle LR schedule:
            - Epoch 0-40: linear increase from min_lr to max_lr
            - Epoch 41-81: linear decrease from max_lr to min_lr
            - Epoch 82-99: linear decrease from min_lr to annihilation_lr
            """
            if epoch < up_epochs:
                # Phase 1: Linear warmup
                progress = epoch / up_epochs
                current_lr = min_lr + (max_lr - min_lr) * progress
                return current_lr / max_lr  # Normalize by max_lr set in optimizer
            elif epoch < up_epochs + down_epochs:
                # Phase 2: Linear cooldown
                progress = (epoch - up_epochs) / down_epochs
                current_lr = max_lr - (max_lr - min_lr) * progress
                return current_lr / max_lr
            else:
                # Phase 3: Annihilation
                progress = (epoch - up_epochs - down_epochs) / annihilate_epochs
                current_lr = min_lr - (min_lr - annihilation_lr) * progress
                return current_lr / max_lr
        
        # Set optimizer base LR to max_lr for proper scaling
        for param_group in optimizer.param_groups:
            param_group['lr'] = max_lr
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing

    # Initialize training state
    start_epoch = 1
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    learning_rates = []
    best_test_acc = 0.0

    # Resume from snapshot if specified
    if args.resume_from:
        start_epoch, train_losses, train_acc, test_losses, test_acc = load_snapshot(
            args.resume_from, model, optimizer, scheduler, device
        )
        start_epoch += 1  # Start from next epoch
        best_test_acc = max(test_acc) if test_acc else 0.0
        print(f"Resuming training from epoch {start_epoch}")

    scaler = GradScaler('cuda', enabled=args.amp)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # No warmup - use scheduler directly
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        print("Starting training...")
        tr_loss, tr_acc = train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            scaler=scaler if args.amp else None,
            use_amp=args.amp,
            max_grad_norm=args.max_grad_norm,
        )
        print("Starting evaluation...")
        te_loss, te_acc = evaluate(model, device, test_loader, criterion, use_amp=args.amp)
        
        # Step scheduler every epoch
        if args.scheduler == "onecycle":
            scheduler.step()  # OneCycleLR steps per batch
        else:
            scheduler.step()  # Other schedulers step per epoch

        train_losses.append(tr_loss)
        train_acc.append(tr_acc)
        test_losses.append(te_loss)
        test_acc.append(te_acc)
        learning_rates.append(current_lr)

        print(
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | "
            f"Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save snapshot based on frequency or best accuracy
        should_save = False
        if args.save_best and te_acc > best_test_acc:
            best_test_acc = te_acc
            should_save = True
            print(f"New best test accuracy: {te_acc:.2f}%")
        elif not args.save_best and epoch % args.snapshot_freq == 0:
            should_save = True

        if should_save:
            save_snapshot(
                model, optimizer, scheduler, epoch, train_losses, train_acc, 
                test_losses, test_acc, args.snapshot_dir, args.model
            )
        
        # Generate plots if requested
        if not args.no_plots and (args.plot_training or args.plot_evaluation):
            if epoch % args.plot_freq == 0 or epoch == args.epochs:
                print(f"\n[Plots] Generating plots for epoch {epoch}...")
                
                # Create plots directory
                os.makedirs(args.plot_dir, exist_ok=True)
                
                # Generate training curves
                if args.plot_training:
                    create_training_summary(
                        train_losses, train_acc, test_losses, test_acc, 
                        learning_rates, args.plot_dir, num_classes=num_classes
                    )
                
                # Generate evaluation plots
                if args.plot_evaluation:
                    create_evaluation_summary(
                        model, device, test_loader, criterion, args.plot_dir, num_classes=num_classes
                    )
                
                print("[OK] Plots generated successfully!")
    
    # ========================================================================
    # FINAL COMPREHENSIVE EVALUATION (ALWAYS RUNS)
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETED - FINAL EVALUATION")
    print("="*70)
    
    # Calculate comprehensive metrics on test set
    print("\n[Evaluation] Computing comprehensive metrics on test set...")
    metrics_calc = MetricsCalculator(num_classes=num_classes)
    eval_results = evaluate_model_comprehensive(model, device, test_loader, criterion, num_classes=num_classes)
    
    metrics = metrics_calc.calculate_metrics(
        eval_results['targets'], 
        eval_results['predictions'],
        eval_results['probabilities']
    )
    
    # Print Training Summary
    print("\n" + "-"*70)
    print("TRAINING SUMMARY")
    print("-"*70)
    print(f"Total Epochs Trained: {args.epochs}")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Training Accuracy: {train_acc[-1]:.2f}%")
    print(f"Best Training Accuracy: {max(train_acc):.2f}% (Epoch {train_acc.index(max(train_acc))+1})")
    print(f"Final Learning Rate: {learning_rates[-1]:.6f}")
    
    # Print Test/Validation Summary
    print("\n" + "-"*70)
    print("TEST/VALIDATION SUMMARY")
    print("-"*70)
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    print(f"Final Test Accuracy: {test_acc[-1]:.2f}%")
    print(f"Best Test Accuracy: {max(test_acc):.2f}% (Epoch {test_acc.index(max(test_acc))+1})")
    
    # Print Overall Metrics
    print("\n" + "-"*70)
    print("COMPREHENSIVE TEST SET METRICS")
    print("-"*70)
    print(f"Top-1 Accuracy (Test): {metrics['accuracy']*100:.2f}%")
    print(f"Top-3 Accuracy (Test): {metrics.get('top_3_accuracy', 0)*100:.2f}%")
    print(f"Top-5 Accuracy (Test): {metrics.get('top_5_accuracy', 0)*100:.2f}%")
    
    print(f"\nMacro-Averaged Metrics:")
    print(f"  - Precision: {metrics['precision_macro']:.4f}")
    print(f"  - Recall:    {metrics['recall_macro']:.4f}")
    print(f"  - F1-Score:  {metrics['f1_macro']:.4f}")
    
    print(f"\nWeighted-Averaged Metrics:")
    print(f"  - Precision: {metrics['precision_weighted']:.4f}")
    print(f"  - Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  - F1-Score:  {metrics['f1_weighted']:.4f}")
    
    # Print Per-Class Metrics
    print("\n" + "-"*70)
    print("PER-CLASS METRICS (TEST SET)")
    print("-"*70)
    
    # For CIFAR-100, show top/bottom performers; for CIFAR-10, show all
    if num_classes == 100:
        print("Top 10 Best Performing Classes:")
        print(f"{'Class':<20} {'Precision':<11} {'Recall':<11} {'F1-Score':<11} {'Support':<8}")
        print("-"*70)
        
        # Sort by F1 score
        f1_scores = metrics['f1_per_class']
        sorted_indices = np.argsort(f1_scores)[::-1]  # descending order
        
        for idx in sorted_indices[:10]:
            class_name = class_names[idx]
            print(f"{class_name:<20} "
                  f"{metrics['precision_per_class'][idx]:<11.4f} "
                  f"{metrics['recall_per_class'][idx]:<11.4f} "
                  f"{metrics['f1_per_class'][idx]:<11.4f} "
                  f"{int(metrics['support_per_class'][idx]):<8}")
        
        print("\nTop 10 Worst Performing Classes:")
        print(f"{'Class':<20} {'Precision':<11} {'Recall':<11} {'F1-Score':<11} {'Support':<8}")
        print("-"*70)
        
        for idx in sorted_indices[-10:]:
            class_name = class_names[idx]
            print(f"{class_name:<20} "
                  f"{metrics['precision_per_class'][idx]:<11.4f} "
                  f"{metrics['recall_per_class'][idx]:<11.4f} "
                  f"{metrics['f1_per_class'][idx]:<11.4f} "
                  f"{int(metrics['support_per_class'][idx]):<8}")
    else:
        # CIFAR-10: Show all classes
        print(f"{'Class':<12} {'Precision':<11} {'Recall':<11} {'F1-Score':<11} {'Support':<8}")
        print("-"*70)
        
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<12} "
                  f"{metrics['precision_per_class'][i]:<11.4f} "
                  f"{metrics['recall_per_class'][i]:<11.4f} "
                  f"{metrics['f1_per_class'][i]:<11.4f} "
                  f"{int(metrics['support_per_class'][i]):<8}")
    
    print("="*70)
    
    # Generate plots if requested
    if not args.no_plots:
        print("\n[Plots] Generating visualizations...")
        os.makedirs(args.plot_dir, exist_ok=True)
        
        # Final training summary plots
        if args.plot_training:
            create_training_summary(
                train_losses, train_acc, test_losses, test_acc, 
                learning_rates, args.plot_dir, num_classes=num_classes
            )
        
        # Final evaluation summary plots (reuse already computed metrics)
        if args.plot_evaluation:
            create_evaluation_summary(
                model, device, test_loader, criterion, args.plot_dir, num_classes=num_classes
            )
        
        print(f"[Output] All plots and reports saved in: {args.plot_dir}")
    
    print("\n[Complete] Training completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
