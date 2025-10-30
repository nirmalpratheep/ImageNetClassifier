"""PyTorch Lightning Module for ImageNet classification."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchmetrics import Accuracy
from model_resnet50 import ResNet50


class CustomThreePhaseLR:
    """Custom learning rate scheduler implementing three-phase schedule:
    Phase 1: max_lr/25 → max_lr over 40% of epochs (warmup/up)
    Phase 2: max_lr → max_lr/100 over 40% of epochs (cool down)
    Phase 3: max_lr/100 → 0.00001 over 20% of epochs (annealing)
    
    Also includes momentum scheduling from momentum_range[0] to momentum_range[1].
    """
    def __init__(self, optimizer, base_lr, max_lr, min_lr, phase1_epochs, phase2_epochs, phase3_epochs, momentum_range=(0.95, 0.85)):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.phase3_epochs = phase3_epochs
        self.total_epochs = phase1_epochs + phase2_epochs + phase3_epochs
        self.last_epoch = -1
        self.momentum_start = momentum_range[0]
        self.momentum_end = momentum_range[1]
    
    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        current_epoch = self.last_epoch
        
        # Determine which phase we're in and calculate LR
        if current_epoch < self.phase1_epochs:
            # Phase 1: linear increase from max_lr/25 to max_lr
            phase1_start_lr = self.max_lr / 25.0
            p = current_epoch / self.phase1_epochs
            lr = phase1_start_lr + p * (self.max_lr - phase1_start_lr)
            self.current_phase = 1
        elif current_epoch < self.phase1_epochs + self.phase2_epochs:
            # Phase 2: linear decrease from max_lr to max_lr/100
            phase2_end_lr = self.max_lr / 100.0
            p = (current_epoch - self.phase1_epochs) / self.phase2_epochs
            lr = self.max_lr - p * (self.max_lr - phase2_end_lr)
            self.current_phase = 2
        else:
            # Phase 3: linear decrease from max_lr/100 to 0.00001
            phase3_start_lr = self.max_lr / 100.0
            phase3_end_lr = 0.00001
            p = (current_epoch - self.phase1_epochs - self.phase2_epochs) / self.phase3_epochs
            lr = phase3_start_lr - p * (phase3_start_lr - phase3_end_lr)
            self.current_phase = 3
        
        # Calculate momentum: linear decrease from momentum_start to momentum_end over all epochs
        momentum_p = current_epoch / self.total_epochs if self.total_epochs > 0 else 0.0
        momentum = self.momentum_start - momentum_p * (self.momentum_start - self.momentum_end)
        
        # Update optimizer parameters
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = momentum
    
    def state_dict(self):
        return {'last_epoch': self.last_epoch}
    
    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']


class ImageNetLightningModule(LightningModule):
    """Lightning Module for ImageNet classification.
    
    Mixed Precision Training (FP16/BF16):
    - Automatically handled by PyTorch Lightning when precision="16-mixed" or "bf16-mixed"
    - Lightning uses torch.cuda.amp internally for automatic mixed precision
    - No manual autocast/GradScaler needed - Lightning handles it
    - Reduces memory usage, enables larger batch sizes, and speeds up training
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        scheduler: str = "onecycle",
        step_size: int = 15,
        gamma: float = 0.1,
        epochs: int = 50,
        label_smoothing: float = 0.1,
        max_grad_norm: float = 1.0,
        scheduler_config: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['scheduler_config'])
        
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.epochs = epochs
        self.label_smoothing = label_smoothing
        self.max_grad_norm = max_grad_norm
        self.scheduler_config = scheduler_config
        
        # Create model
        self.model = ResNet50(num_classes=num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Metrics (these will be logged to TensorBoard)
        # Accuracy is returned as 0-1, we'll format as percentage in logging
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        
        # Track metrics for logging
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []
        self.best_val_acc = 0.0
        
        # Custom scheduler instance (for onecycle)
        self.custom_scheduler = None
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        # Update accuracy metric (for epoch-level aggregation)
        self.train_accuracy(logits, y)
        
        # Compute batch-level accuracy for per-step logging
        # Use detach() to ensure we're computing on the actual predictions
        preds = logits.detach().argmax(dim=-1)
        correct = (preds == y).float()
        
        # Calculate accuracy - ensure we have actual tensor values
        # For distributed training, compute accuracy before any sync
        num_correct = correct.sum().item()
        total_samples = len(y)
        batch_acc = (num_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
        
        # Log metrics to TensorBoard
        # Train loss: logged both per-step and per-epoch
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Train accuracy: batch-level for per-step (shown in progress bar)
        # Log as a float value, not tensor
        self.log('train_acc_step', batch_acc, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=False)
        # Train accuracy: epoch-level uses the accumulated metric (sync across GPUs)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        # Update accuracy metric
        self.val_accuracy(logits, y)
        
        # Log metrics to TensorBoard
        # Val loss: logged per-epoch
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Val accuracy: logged per-epoch (automatically aggregated)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Only print on main process (rank 0) to avoid duplication with DDP
        if self.trainer.global_rank == 0:
            current_lr = self.optimizers().param_groups[0]['lr']
            current_momentum = self.optimizers().param_groups[0].get('momentum', self.momentum)
            current_epoch = self.current_epoch
            total_epochs = self.trainer.max_epochs
            print(f"\n{'='*70}")
            print(f"Epoch {current_epoch + 1}/{total_epochs} - Learning Rate: {current_lr:.6f} - Momentum: {current_momentum:.4f}")
            print(f"{'='*70}")
    
    def on_train_epoch_end(self):
        """Collect epoch-level metrics and print epoch summary."""
        # Get metrics from logger or compute from logged values
        epoch_loss = self.trainer.callback_metrics.get('train_loss_epoch', torch.tensor(0.0))
        # train_acc is logged as the metric object, Lightning adds _epoch suffix
        epoch_acc_tensor = self.trainer.callback_metrics.get('train_acc', None)
        if epoch_acc_tensor is None:
            # Fallback: compute from the metric object directly
            epoch_acc = self.train_accuracy.compute()
        else:
            epoch_acc = epoch_acc_tensor if isinstance(epoch_acc_tensor, torch.Tensor) else torch.tensor(epoch_acc_tensor)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        current_momentum = self.optimizers().param_groups[0].get('momentum', self.momentum)
        
        # Convert to CPU float for storage
        epoch_loss_val = epoch_loss.item() if isinstance(epoch_loss, torch.Tensor) else float(epoch_loss)
        epoch_acc_val = epoch_acc.item() * 100.0 if isinstance(epoch_acc, torch.Tensor) else float(epoch_acc) * 100.0
        
        self.train_losses.append(epoch_loss_val)
        self.train_accs.append(epoch_acc_val)
        self.learning_rates.append(current_lr)
        
        # Print epoch summary - only on main process (rank 0) to avoid duplication
        if self.trainer.global_rank == 0:
            current_epoch = self.current_epoch
            print(f"\n{'='*70}")
            print(f"Epoch {current_epoch + 1} Training Summary:")
            print(f"  Train Loss: {epoch_loss_val:.4f}")
            print(f"  Train Accuracy: {epoch_acc_val:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Momentum: {current_momentum:.4f}")
            print(f"{'='*70}")
        
        # Reset metrics for next epoch
        self.train_accuracy.reset()
    
    def on_validation_epoch_end(self):
        """Collect validation metrics and print validation summary."""
        epoch_loss = self.trainer.callback_metrics.get('val_loss_epoch', torch.tensor(0.0))
        epoch_acc_tensor = self.trainer.callback_metrics.get('val_acc', None)
        if epoch_acc_tensor is None:
            epoch_acc = self.val_accuracy.compute()
        else:
            epoch_acc = epoch_acc_tensor if isinstance(epoch_acc_tensor, torch.Tensor) else torch.tensor(epoch_acc_tensor)
        
        epoch_loss_val = epoch_loss.item() if isinstance(epoch_loss, torch.Tensor) else float(epoch_loss)
        val_acc_percent = epoch_acc.item() * 100.0 if isinstance(epoch_acc, torch.Tensor) else float(epoch_acc) * 100.0
        
        self.val_losses.append(epoch_loss_val)
        self.val_accs.append(val_acc_percent)
        
        # Track best validation accuracy
        if val_acc_percent > self.best_val_acc:
            self.best_val_acc = val_acc_percent
        
        # Print validation summary - only on main process (rank 0) to avoid duplication
        if self.trainer.global_rank == 0:
            current_epoch = self.current_epoch
            print(f"\n{'='*70}")
            print(f"Epoch {current_epoch + 1} Validation Summary:")
            print(f"  Val Loss: {epoch_loss_val:.4f}")
            print(f"  Val Accuracy: {val_acc_percent:.2f}%")
            if val_acc_percent == self.best_val_acc:
                print(f"  ★ New best validation accuracy!")
            print(f"  Best Val Accuracy: {self.best_val_acc:.2f}%")
            print(f"{'='*70}")
        
        # Reset metrics for next epoch
        self.val_accuracy.reset()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True
        )
        
        # Use saved scheduler config if resuming, otherwise create from hyperparameters
        if self.scheduler_config and self.scheduler_config.get('type') == self.scheduler_type:
            scheduler = self._create_scheduler_from_config(optimizer, self.scheduler_config)
        else:
            scheduler = self._create_scheduler(optimizer)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    def _create_scheduler(self, optimizer):
        """Create scheduler from hyperparameters."""
        current_lr = self.lr  # May have been updated by LR finder
        
        if self.scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        elif self.scheduler_type == "step":
            return StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        elif self.scheduler_type == "onecycle":
            # Custom three-phase schedule
            max_lr = current_lr
            base_lr = current_lr / 10.0
            min_lr = current_lr / 100.0
            
            phase1_epochs = max(1, int(self.epochs * 0.4))
            phase2_epochs = max(1, int(self.epochs * 0.4))
            phase3_epochs = max(1, self.epochs - phase1_epochs - phase2_epochs)
            
            self.custom_scheduler = CustomThreePhaseLR(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                min_lr=min_lr,
                phase1_epochs=phase1_epochs,
                phase2_epochs=phase2_epochs,
                phase3_epochs=phase3_epochs,
                momentum_range=(0.95, 0.85)
            )
            
            # Initialize scheduler
            self.custom_scheduler.step()
            
            return self.custom_scheduler
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_type}")
    
    def _create_scheduler_from_config(self, optimizer, config):
        """Create scheduler from saved configuration."""
        if config['type'] == "cosine":
            return CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config.get('eta_min', 1e-6))
        elif config['type'] == "step":
            return StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
        elif config['type'] == "onecycle":
            momentum_range = config.get('momentum_range', (0.95, 0.85))
            self.custom_scheduler = CustomThreePhaseLR(
                optimizer,
                base_lr=config['base_lr'],
                max_lr=config['max_lr'],
                min_lr=config['min_lr'],
                phase1_epochs=config['phase1_epochs'],
                phase2_epochs=config['phase2_epochs'],
                phase3_epochs=config['phase3_epochs'],
                momentum_range=momentum_range
            )
            return self.custom_scheduler
        else:
            raise ValueError(f"Unknown scheduler type: {config['type']}")
    
    def get_scheduler_config(self):
        """Get current scheduler configuration for saving."""
        scheduler = self.lr_schedulers()
        if scheduler is None:
            return None
        
        config = {'type': self.scheduler_type}
        
        if self.scheduler_type == "cosine":
            config['T_max'] = scheduler.T_max
            config['eta_min'] = scheduler.eta_min
        elif self.scheduler_type == "step":
            config['step_size'] = scheduler.step_size
            config['gamma'] = scheduler.gamma
        elif self.scheduler_type == "onecycle":
            config['base_lr'] = scheduler.base_lr
            config['max_lr'] = scheduler.max_lr
            config['min_lr'] = scheduler.min_lr
            config['phase1_epochs'] = scheduler.phase1_epochs
            config['phase2_epochs'] = scheduler.phase2_epochs
            config['phase3_epochs'] = scheduler.phase3_epochs
            config['momentum_range'] = (scheduler.momentum_start, scheduler.momentum_end)
        
        return config
    
    def lr_scheduler_step(self, scheduler, metric):
        """Custom LR scheduler step to support CustomThreePhaseLR."""
        if isinstance(scheduler, CustomThreePhaseLR):
            scheduler.step()
        else:
            scheduler.step()
    
    def on_train_start(self):
        """Called at the start of training."""
        if self.custom_scheduler is None and self.scheduler_type == "onecycle":
            # Initialize custom scheduler if not already done
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, CustomThreePhaseLR):
                self.custom_scheduler = scheduler
