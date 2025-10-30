"""Main training script using PyTorch Lightning with TensorBoard logging."""
import argparse
import os
import sys
import logging
from datetime import datetime
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from lightning_model import ImageNetLightningModule
from data_module import ImageNetDataModule
from lr_finder import find_lr


class TeeLogger:
    """Logger that writes to both file and stdout."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        sys.stdout = self.stdout
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ImageNet Training with PyTorch Lightning")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    
    # Scheduler arguments
    parser.add_argument("--scheduler", type=str, default="onecycle", choices=["cosine", "step", "onecycle"],
                       help="Learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=15, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers (recommended: 4-16 based on CPU cores)")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches prefetched per worker (default: 2)")
    parser.add_argument("--subset_size", type=int, default=None, help="Limit number of classes")
    parser.add_argument("--max_samples_per_class", type=int, default=None, help="Max samples per class")
    parser.add_argument("--augmentation", action="store_true", default=True, help="Enable data augmentation")
    
    # LR Finder arguments
    parser.add_argument("--find_lr", action="store_true", help="Run learning rate finder")
    parser.add_argument("--lr_start", type=float, default=1e-7, help="LR finder start LR")
    parser.add_argument("--lr_end", type=float, default=10, help="LR finder end LR")
    parser.add_argument("--lr_iter", type=int, default=100, help="LR finder iterations")
    parser.add_argument("--lr_plot", type=str, default="./lr_finder_plot.png", help="LR finder plot path")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--save_top_k", type=int, default=3, help="Save top K checkpoints")
    parser.add_argument("--monitor", type=str, default="val_acc", help="Metric to monitor for best checkpoint")
    parser.add_argument("--mode", type=str, default="max", choices=["min", "max"], help="Monitor mode")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard log directory")
    parser.add_argument("--name", type=str, default="imagenet_resnet50", help="Experiment name")
    parser.add_argument("--version", type=str, default=None, help="Experiment version")
    parser.add_argument("--stdout_file", type=str, default=None, help="File to save stdout (default: log_dir/name/version/stdout.log)")
    parser.add_argument("--save_stdout", action="store_true", default=True, help="Save all stdout to file")
    
    # Training features
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (deprecated: use --precision 16)")
    parser.add_argument("--precision", type=str, default="16", choices=["16", "32", "bf16"],
                       help="Precision (16=FP16 mixed precision, 32=FP32, bf16=BF16). Default: 16 (FP16 mixed precision enabled)")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision, use FP32 (sets precision to 32)")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, 
                       help="Number of batches to accumulate before optimizer step (effective batch size = batch_size × accumulate_grad_batches × num_gpus)")
    
    # Device arguments
    parser.add_argument("--devices", type=int, default=None, help="Number of GPUs (None = auto-detect all)")
    parser.add_argument("--use_multi_gpu", action="store_true", help="Force use of all available GPUs (same as --devices with auto-detect)")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy (ddp, ddp_spawn, etc.)")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def setup_logging(log_dir, name, version, stdout_file=None, save_stdout=True):
    """Setup logging to file and console."""
    log_path = None
    
    if save_stdout:
        if stdout_file is None:
            # Auto-generate log file path
            if version:
                log_dir_full = os.path.join(log_dir, name, version)
            else:
                # Create versioned directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir_full = os.path.join(log_dir, name, f"version_{timestamp}")
                version = f"version_{timestamp}"
            
            os.makedirs(log_dir_full, exist_ok=True)
            log_path = os.path.join(log_dir_full, "stdout.log")
        else:
            log_path = stdout_file
            os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        
        print(f"All stdout will be saved to: {log_path}")
        
        # Setup Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return TeeLogger(log_path), log_path
    else:
        # Just use console logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        return None, None


def main():
    args = parse_args()
    
    # Setup stdout logging first (before any prints)
    tee_logger, log_file = setup_logging(
        args.log_dir, 
        args.name, 
        args.version, 
        args.stdout_file,
        args.save_stdout
    )
    
    try:
        # Set random seed
        seed_everything(args.seed, workers=True)
        
        # Setup device and multi-GPU
        if args.use_multi_gpu:
            # Force use all available GPUs
            if torch.cuda.is_available():
                args.devices = torch.cuda.device_count()
                print(f"--use_multi_gpu specified: using all {args.devices} GPU(s)")
            else:
                print("Warning: --use_multi_gpu specified but CUDA not available")
                args.devices = 1
        
        if args.devices is None:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                args.devices = num_gpus
                print(f"Auto-detected {num_gpus} GPU(s)")
            else:
                args.devices = 1
                print("CUDA not available, using CPU")
        
        # Verify GPU availability
        if torch.cuda.is_available():
            num_gpus_available = torch.cuda.device_count()
            print(f"CUDA available: {num_gpus_available} GPU(s) detected")
            
            # Check for CUDA_VISIBLE_DEVICES restriction
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible is not None:
                print(f"⚠️  WARNING: CUDA_VISIBLE_DEVICES is set to: {cuda_visible}")
                print("  This WILL restrict which GPUs are available to PyTorch!")
                print("  For multi-GPU training, you should unset this variable:")
                print("    unset CUDA_VISIBLE_DEVICES")
                print("  Or set it to use all GPUs:")
                print("    export CUDA_VISIBLE_DEVICES=0,1,2,3")
            
            # Clear any existing GPU memory/cache
            torch.cuda.empty_cache()
            
            # Check GPU memory usage
            for i in range(num_gpus_available):
                gpu_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                print(f"  GPU {i}: {gpu_name}")
                if memory_allocated > 0.1 or memory_reserved > 0.1:
                    print(f"    ⚠️  Warning: GPU {i} has {memory_reserved:.2f} GB memory allocated")
                    print(f"       You may want to clear it with: torch.cuda.empty_cache() or restart Python")
        
        # Convert to list format for explicit GPU specification
        if torch.cuda.is_available() and args.devices > 1:
            device_list = list(range(args.devices))
            print(f"Configuring trainer to use GPUs: {device_list}")
        else:
            device_list = args.devices
        
        print(f"Final configuration: using {args.devices} device(s)")
        
        # Create data module
        data_module = ImageNetDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            subset_size=args.subset_size,
            augmentation=args.augmentation,
            max_samples_per_class=args.max_samples_per_class,
        )
        
        # Setup data to get num_classes
        data_module.setup("fit")
        num_classes = data_module.num_classes
        
        # Print DataLoader optimization info
        print(f"\nDataLoader Optimization:")
        print(f"  num_workers: {args.num_workers} (recommended: 4-16 based on CPU cores)")
        print(f"  pin_memory: True (faster GPU transfer)")
        print(f"  prefetch_factor: {args.prefetch_factor} (batches prefetched per worker)")
        print(f"  persistent_workers: {args.num_workers > 0} (keep workers alive between epochs)")
        if args.devices > 1:
            print(f"  DistributedSampler: Enabled automatically by Lightning for DDP")
            print(f"    - Each GPU sees unique, non-overlapping batches")
        
        print(f"Dataset: ImageNet ({num_classes} classes)")
        
        # Create model
        model = ImageNetLightningModule(
            num_classes=num_classes,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            step_size=args.step_size,
            gamma=args.gamma,
            epochs=args.epochs,
            label_smoothing=args.label_smoothing,
            max_grad_norm=args.max_grad_norm,
        )
        
        # Compile model for PyTorch 2.x speedup (20-30% faster)
        # Note: Cannot use torch.compile() with DDP/multiprocessing due to pickle issues
        try:
            if hasattr(torch, 'compile') and torch.cuda.is_available() and args.devices <= 1:
                model = torch.compile(model, mode='reduce-overhead')
                print("Model compiled with torch.compile (PyTorch 2.x speedup enabled)")
            elif args.devices > 1:
                print("torch.compile() disabled for multi-GPU training (not compatible with DDP)")
        except Exception as e:
            print(f"Could not compile model: {e}")
        
        # Run LR finder if requested (before creating trainer)
        if args.find_lr:
            print("\n" + "="*70)
            print("RUNNING LEARNING RATE FINDER")
            print("="*70)
            
            try:
                # Get the underlying PyTorch model
                pytorch_model = model.model
                
                suggested_lr, fig = find_lr(
                    model=pytorch_model,
                    train_loader=data_module.train_dataloader(),
                    optimizer=torch.optim.SGD(
                        pytorch_model.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=True
                    ),
                    criterion=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    start_lr=args.lr_start,
                    end_lr=args.lr_end,
                    num_iter=args.lr_iter,
                    plot=True,
                    save_path=args.lr_plot,
                    use_amp=args.amp
                )
                
                print(f"\nSuggested learning rate: {suggested_lr:.2e}")
                print(f"LR finder plot saved to: {args.lr_plot}")
                
                # Update model learning rate
                model.lr = suggested_lr
                model.hparams.lr = suggested_lr  # Also update hyperparameters
                print(f"Updated model learning rate to: {suggested_lr:.2e}")
                
                # Move model back to CPU to ensure clean state for multi-GPU training
                # PyTorch Lightning will handle device placement during training
                if hasattr(model.model, 'cpu'):
                    model.model = model.model.cpu()
                # Clear CUDA cache to free up GPU memory
                torch.cuda.empty_cache()
                print("Model moved to CPU and GPU cache cleared (trainer will handle multi-GPU setup)")
                
            except Exception as e:
                print(f"LR finder failed: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing with original learning rate...")
            
            print("="*70 + "\n")
        
        # Setup TensorBoard logger
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=args.name,
            version=args.version,
        )
        
        print(f"TensorBoard logs will be saved to: {logger.log_dir}")
        print(f"View with: tensorboard --logdir {args.log_dir}")
    
        # Setup checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}",
            monitor=args.monitor,
            mode=args.mode,
            save_top_k=args.save_top_k,
            save_last=True,
            every_n_epochs=1,  # Save checkpoint after every epoch
            save_on_train_epoch_end=False,  # Save after validation
        )
        
        # Learning rate monitor callback (logs LR to TensorBoard)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Determine precision (FP16 mixed precision enabled by default)
        # PyTorch Lightning uses torch.cuda.amp automatically when precision="16-mixed"
        if args.no_amp:
            # Force FP32 if --no_amp is specified
            precision = "32"
            print("Mixed precision DISABLED: Using FP32 (--no_amp flag set)")
        elif args.amp:
            # Legacy --amp flag support
            precision = "16-mixed" if torch.cuda.is_available() else "32"
            print("Mixed precision enabled via --amp flag")
        else:
            # Default: use precision argument (defaults to "16" for FP16 mixed precision)
            if args.precision == "32":
                precision = "32"
            else:
                # Use mixed precision (16-mixed or bf16-mixed)
                precision = f"{args.precision}-mixed"
        
        if precision in ["16-mixed", "bf16-mixed"] and torch.cuda.is_available():
            print(f"Mixed Precision Training (FP16/BF16) enabled via torch.cuda.amp")
            print(f"  - Reduces memory usage → allows larger batch sizes → faster training")
        
        # Setup training strategy for multi-GPU
        # For PyTorch Lightning, use integer for devices (works better than list)
        # Lightning will automatically use devices 0 through args.devices-1
        devices_param = args.devices  # Use integer, not list
        
        # Determine accelerator - must be "gpu" for multi-GPU, not "auto"
        if args.devices > 1 and torch.cuda.is_available():
            accelerator = "gpu"
        elif args.accelerator == "auto":
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        else:
            accelerator = args.accelerator
        
        # Configure Rich progress bar with text wrapping and visible output
        try:
            # RichProgressBar with proper configuration
            progress_bar = RichProgressBar(
                leave=True,  # Keep progress bar visible after completion
                refresh_rate=10,  # Update every 10 steps
                console_kwargs={"width": None, "force_terminal": True, "force_width": None},  # Auto-detect width, enable wrapping
            )
            callbacks = [checkpoint_callback, lr_monitor, progress_bar]
        except Exception:
            # Fallback if RichProgressBar is not available
            progress_bar = None
            callbacks = [checkpoint_callback, lr_monitor]
        
        trainer_kwargs = {
            'max_epochs': args.epochs,
            'accelerator': accelerator,
            'devices': devices_param,
            'precision': precision,
            'logger': logger,
            'callbacks': callbacks,
            'gradient_clip_val': args.gradient_clip_val,
            'gradient_clip_algorithm': "norm",
            'accumulate_grad_batches': args.accumulate_grad_batches,
            'log_every_n_steps': 50,
            'val_check_interval': 1.0,  # Validate every epoch
            'enable_progress_bar': True,
            'enable_model_summary': True,
            'deterministic': False,  # Set to True for reproducibility (slower)
            'benchmark': True,
        }
        
        # Setup strategy - only add if needed (don't pass None)
        # Use "ddp_spawn" for better process isolation - spawns clean processes
        strategy_display = "auto (default)"
        if args.strategy == "auto":
            if args.devices > 1:
                # Use "ddp_spawn" which explicitly spawns processes and isolates GPUs better
                trainer_kwargs['strategy'] = "ddp_spawn"
                strategy_display = "ddp_spawn"
                print(f"Multi-GPU detected: Using DDP Spawn strategy with {args.devices} GPUs")
                print(f"  Devices: {devices_param}, Accelerator: {accelerator}")
                print(f"  Strategy: 'ddp_spawn' (PyTorch Lightning will spawn {args.devices} processes, one per GPU)")
            # If single GPU/CPU, don't add strategy parameter (use Lightning default)
        elif args.strategy != "auto":
            # Pass through the strategy string directly
            trainer_kwargs['strategy'] = args.strategy
            strategy_display = args.strategy
        
        # Calculate effective batch size
        effective_batch_size = args.batch_size * args.accumulate_grad_batches * args.devices
        print(f"\nBatch Size Configuration:")
        print(f"  Per-GPU batch size: {args.batch_size}")
        print(f"  Gradient accumulation: {args.accumulate_grad_batches} batches")
        print(f"  Number of GPUs: {args.devices}")
        print(f"  Effective batch size: {effective_batch_size} (batch_size × accumulate_grad_batches × num_gpus)")
        
        # Print final configuration for debugging
        print(f"\nTrainer Configuration:")
        print(f"  Accelerator: {accelerator}")
        print(f"  Devices: {devices_param}")
        print(f"  Strategy: {strategy_display}")
        print(f"  Precision: {precision}")
        if precision in ["16-mixed", "bf16-mixed"]:
            print(f"    → Using torch.cuda.amp for automatic mixed precision")
            print(f"    → Memory efficient: enables larger batch sizes")
        print(f"  Gradient accumulation: {args.accumulate_grad_batches} batches")
        print(f"  Gradient clipping: {args.gradient_clip_val} (norm)")
        
        # Create trainer
        trainer = Trainer(**trainer_kwargs)
        
        # Verify trainer configuration
        print(f"\nTrainer Created Successfully:")
        print(f"  Number of devices: {getattr(trainer, 'num_devices', 'N/A')}")
        print(f"  Strategy: {type(trainer.strategy).__name__}")
        print(f"  Strategy class: {trainer.strategy.__class__.__name__}")
        
        # Try to get process/world size info if available
        try:
            num_processes = getattr(trainer, 'num_processes', None)
            if num_processes is not None:
                print(f"  Number of processes: {num_processes}")
        except AttributeError:
            pass
        
        try:
            world_size = getattr(trainer.strategy, 'world_size', None)
            if world_size is not None:
                print(f"  World size: {world_size}")
        except AttributeError:
            pass
        
        # Check PyTorch Lightning version
        import pytorch_lightning as pl
        print(f"  PyTorch Lightning version: {pl.__version__}")
        
        # Important: With DDP, the actual training will spawn separate processes
        # Each process will use one GPU. Check nvidia-smi AFTER training starts.
        if args.devices > 1:
            print(f"\n⚠️  IMPORTANT: With DDP and {args.devices} devices, PyTorch Lightning will spawn {args.devices} processes.")
            print(f"   You should see {args.devices} Python processes in nvidia-smi once training starts.")
            print(f"   If you only see 1 process, there may be a configuration issue.")
        
        # Start training
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Model: ResNet-50")
        print(f"Classes: {num_classes}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * args.devices}")
        print(f"Learning rate: {model.lr:.6f}")
        print(f"Scheduler: {args.scheduler}")
        print(f"Epochs: {args.epochs}")
        print(f"Precision: {precision}")
        print(f"Strategy: {strategy_display}")
        print("="*70 + "\n")
        
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=args.resume_from,
        )
    
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Latest checkpoint: {checkpoint_callback.last_model_path}")
        print(f"TensorBoard logs: {logger.log_dir}")
        print(f"To view logs: tensorboard --logdir {args.log_dir}")
        if log_file:
            print(f"Stdout log: {log_file}")
        print("\nTo resume training from this checkpoint, use:")
        print(f"  python main_lightning.py --resume_from {checkpoint_callback.last_model_path} <other_args>")
        print("="*70)
    finally:
        # Close tee logger if it exists
        if tee_logger:
            tee_logger.close()
            print(f"Stdout log saved to: {log_file}")


if __name__ == "__main__":
    main()

