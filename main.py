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

# Visualization imports
from visualization import (
    create_training_summary, create_evaluation_summary, 
    evaluate_model_comprehensive, TrainingVisualizer, MetricsCalculator,
    CIFAR10_CLASSES, CIFAR100_CLASSES
)

# Import wandb for experiment tracking (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


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


def build_model(device: torch.device, num_classes: int = 1000):
    """Build ResNet-50 model for ImageNet-1K."""
    from model_resnet50 import ResNet50
    
    model = ResNet50(num_classes=num_classes)
    return model.to(device)


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
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--step_size", type=int, default=15, help="Step size for StepLR scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs (disabled)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "onecycle"], help="Learning rate scheduler")
    
    # OneCycleLR specific arguments
    parser.add_argument("--onecycle_pct_start", type=float, default=0.3, help="OneCycleLR: percent of cycle for warmup (default: 0.3)")
    parser.add_argument("--onecycle_div_factor", type=float, default=25.0, help="OneCycleLR: initial_lr = max_lr/div_factor (default: 25)")
    parser.add_argument("--onecycle_final_div_factor", type=float, default=10000.0, help="OneCycleLR: min_lr = initial_lr/final_div_factor (default: 10000)")
    parser.add_argument("--onecycle_three_phase", action="store_true", help="OneCycleLR: use three-phase schedule (default: False)")
    parser.add_argument("--onecycle_anneal_strategy", type=str, default="cos", choices=["cos", "linear"], help="OneCycleLR: annealing strategy (default: cos)")
    
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    # Dataset arguments
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use (for testing/debugging)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio for single directory datasets (default: 0.2)")
    
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
    
    # Wandb experiment tracking arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="imagenet-classification", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name (auto-generated if not specified)")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="Wandb tags (space-separated)")
    parser.add_argument("--wandb_group", type=str, default=None, help="Wandb group name for organizing related runs")
    parser.add_argument("--wandb_notes", type=str, default=None, help="Notes/description for the wandb run")

    args = parser.parse_args()
    set_seed(42)
    device = get_device(prefer_cuda=not args.no_cuda)

    # Optimize data loading based on device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    pin_memory = use_cuda  # Use pin_memory when CUDA is available for better performance
    num_workers = min(args.num_workers, 2) if not use_cuda else args.num_workers
    
    print(f"Data loading: num_workers={num_workers}, pin_memory={pin_memory}, use_cuda={use_cuda}")
    
    print(f"Loading ImageNet-1K dataset from {args.data_dir}...")
    if args.max_samples:
        print(f"Limited to {args.max_samples} samples per split")
    
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle_train=True,
        streaming=False,  # Always use offline data
        max_samples=args.max_samples,
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
    )
    
    # Test data loading and detect number of classes
    print("Testing data loading...")
    try:
        test_batch = next(iter(train_loader))
        print(f"Data loading successful! Batch shape: {test_batch[0].shape}, labels: {test_batch[1].shape}")
        
        # Detect number of classes from the dataset
        if hasattr(train_loader.dataset, 'classes'):
            num_classes = len(train_loader.dataset.classes)
        elif hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'classes'):
            # For Subset datasets
            num_classes = len(train_loader.dataset.dataset.classes)
        else:
            # Try to infer from data
            max_label = test_batch[1].max().item()
            num_classes = max_label + 1
            print(f"⚠️  Could not detect classes from dataset, inferred {num_classes} classes from max label")
        
        print(f"✓ Detected {num_classes} classes in dataset")
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        return 1

    model = build_model(device, num_classes=num_classes)
    print(f"Device: {device}")
    print(f"Model: ResNet-50 for {num_classes} classes")
    print(f"Model loaded, starting training...")
    
    # Dataset configuration  
    dataset_name = f"ImageNet-{num_classes}K" if num_classes == 1000 else f"ImageNet-{num_classes}-classes"
    input_size = 224
    
    print(f"Dataset: {dataset_name} ({num_classes} classes)")
    
    # Define class names for ImageNet-1K
    class_names = [f"class_{i}" for i in range(num_classes)]
    
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
                    save_path=args.lr_plot,
                    use_wandb=args.use_wandb,
                    wandb_run_name=args.wandb_run_name,
                    wandb_project=args.wandb_project,
                    wandb_tags=args.wandb_tags
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
                    use_amp=args.amp,
                    use_wandb=args.use_wandb,
                    wandb_run_name=args.wandb_run_name,
                    wandb_project=args.wandb_project,
                    wandb_tags=args.wandb_tags
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
        # Use PyTorch's official OneCycleLR scheduler
        # Reference: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        # Based on the paper "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
        
        # Calculate steps per epoch
        try:
            steps_per_epoch = len(train_loader)
        except (TypeError, ValueError):
            # For streaming dataloaders, estimate based on dataset size and batch size
            if args.max_samples:
                steps_per_epoch = (args.max_samples + args.batch_size - 1) // args.batch_size
            else:
                steps_per_epoch = 1000  # Default estimate
        
        initial_lr = args.lr / args.onecycle_div_factor
        min_lr = initial_lr / args.onecycle_final_div_factor
        
        print(f"\n[OneCycleLR] PyTorch Official Implementation:")
        print(f"   - Max LR: {args.lr:.4f}")
        print(f"   - Epochs: {args.epochs}")
        print(f"   - Steps per epoch: {steps_per_epoch}")
        print(f"   - Total steps: {args.epochs * steps_per_epoch}")
        print(f"   - Initial LR: {initial_lr:.6f} (max_lr / {args.onecycle_div_factor})")
        print(f"   - Min LR: {min_lr:.8f} (initial_lr / {args.onecycle_final_div_factor})")
        print(f"   - Anneal strategy: {args.onecycle_anneal_strategy}")
        print(f"   - Percent start: {args.onecycle_pct_start*100:.0f}% (warmup phase)")
        print(f"   - Three phase: {args.onecycle_three_phase}")
        print(f"   - Cycle momentum: True (0.85 <-> 0.95)\n")
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.onecycle_pct_start,
            anneal_strategy=args.onecycle_anneal_strategy,
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
            three_phase=args.onecycle_three_phase
        )
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
    
    # Initialize wandb for main training (only if not running LR finder)
    if args.use_wandb and WANDB_AVAILABLE and not args.find_lr:
        # Generate run name if not provided
        run_name = args.wandb_run_name or f"imagenet_resnet50_{args.scheduler}_lr_{args.lr:.0e}_bs_{args.batch_size}"
        
        # Prepare config
        wandb_config = {
            "model": "ResNet-50 v1.5",
            "dataset": "ImageNet-1K",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "optimizer": "SGD",
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "device": str(device),
            "amp_enabled": args.amp,
            "max_grad_norm": args.max_grad_norm,
            "num_classes": num_classes,
            "input_size": input_size
        }
        
        # Add scheduler-specific config
        if args.scheduler == "onecycle":
            wandb_config.update({
                "onecycle_pct_start": args.onecycle_pct_start,
                "onecycle_div_factor": args.onecycle_div_factor,
                "onecycle_final_div_factor": args.onecycle_final_div_factor,
                "onecycle_anneal_strategy": args.onecycle_anneal_strategy,
                "onecycle_three_phase": args.onecycle_three_phase
            })
        elif args.scheduler == "step":
            wandb_config.update({
                "step_size": args.step_size,
                "gamma": args.gamma
            })
        
        # Prepare tags
        tags = args.wandb_tags or ["imagenet", "resnet50", "training"]
        if args.amp:
            tags.append("mixed_precision")
        tags.append(args.scheduler)
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=tags,
            config=wandb_config,
            group=args.wandb_group,
            notes=args.wandb_notes,
            reinit=True
        )
        
        # Log model summary
        wandb.watch(model, log_freq=100, log_graph=False)
        print(f"📊 Wandb training logging initialized: {args.wandb_project}/{run_name}")
        
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("⚠️  Warning: Wandb requested but not available. Install with: pip install wandb")

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
            scheduler=scheduler if args.scheduler == "onecycle" else None,
            scheduler_step_per_batch=(args.scheduler == "onecycle"),
            epoch=epoch,
            use_wandb=args.use_wandb and not args.find_lr,
            log_freq=100
        )
        print("Starting evaluation...")
        te_loss, te_acc = evaluate(
            model, 
            device, 
            test_loader, 
            criterion, 
            use_amp=args.amp,
            epoch=epoch,
            use_wandb=args.use_wandb and not args.find_lr
        )
        
        # Step scheduler per epoch (except OneCycleLR which steps per batch)
        if args.scheduler != "onecycle":
            scheduler.step()

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
        
        # Log metrics to wandb
        if args.use_wandb and WANDB_AVAILABLE and not args.find_lr:
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/accuracy": tr_acc,
                "val/loss": te_loss,
                "val/accuracy": te_acc,
                "learning_rate": current_lr,
                "best_val_accuracy": best_test_acc
            }, step=epoch)

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
                test_losses, test_acc, args.snapshot_dir, "resnet50"
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
        actual_num_classes = len(f1_scores)
        sorted_indices = np.argsort(f1_scores)[::-1]  # descending order
        
        print(f"Showing top/bottom performers from {actual_num_classes} classes with data:")
        print()
        
        # Show top performers (limited by actual classes available)
        top_count = min(10, actual_num_classes)
        for idx in sorted_indices[:top_count]:
            # Ensure we don't exceed class_names bounds
            class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            print(f"{class_name:<20} "
                  f"{metrics['precision_per_class'][idx]:<11.4f} "
                  f"{metrics['recall_per_class'][idx]:<11.4f} "
                  f"{metrics['f1_per_class'][idx]:<11.4f} "
                  f"{int(metrics['support_per_class'][idx]):<8}")
        
        print(f"\nTop {min(10, actual_num_classes)} Worst Performing Classes:")
        print(f"{'Class':<20} {'Precision':<11} {'Recall':<11} {'F1-Score':<11} {'Support':<8}")
        print("-"*70)
        
        # Show bottom performers (limited by actual classes available)
        bottom_count = min(10, actual_num_classes)
        for idx in sorted_indices[-bottom_count:]:
            # Ensure we don't exceed class_names bounds
            class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            print(f"{class_name:<20} "
                  f"{metrics['precision_per_class'][idx]:<11.4f} "
                  f"{metrics['recall_per_class'][idx]:<11.4f} "
                  f"{metrics['f1_per_class'][idx]:<11.4f} "
                  f"{int(metrics['support_per_class'][idx]):<8}")
    else:
        # Show metrics for classes that actually have data (handle partial sampling)
        actual_num_classes = len(metrics['precision_per_class'])
        print(f"{'Class':<12} {'Precision':<11} {'Recall':<11} {'F1-Score':<11} {'Support':<8}")
        print("-"*70)
        print(f"Showing metrics for {actual_num_classes} classes (out of {num_classes} total classes)")
        print("-"*70)
        
        # Only show metrics for classes that have data
        for i in range(min(actual_num_classes, len(class_names))):
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
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
    
    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE and not args.find_lr:
        # Log final metrics summary
        wandb.log({
            "final/train_loss": train_losses[-1],
            "final/train_accuracy": train_acc[-1],
            "final/val_loss": test_losses[-1],
            "final/val_accuracy": test_acc[-1],
            "final/best_val_accuracy": max(test_acc),
            "final/total_epochs": args.epochs
        })
        wandb.finish()
        print("📊 Wandb run completed successfully!")
    
    print("\n[Complete] Training completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
