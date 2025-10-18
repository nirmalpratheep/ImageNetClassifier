import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

# Import wandb for experiment tracking (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def train_epoch(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
    max_grad_norm: float = 1.0,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
    epoch: int = None,
    use_wandb: bool = False,
    log_freq: int = 100
):
    """Train the model for one epoch with optional AMP and gradient clipping."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # Step scheduler per batch if requested (for OneCycleLR)
        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Log to wandb periodically
        if use_wandb and WANDB_AVAILABLE and batch_idx % log_freq == 0:
            batch_acc = 100. * predicted.eq(target).sum().item() / target.size(0)
            current_lr = optimizer.param_groups[0]['lr']
            
            log_data = {
                "batch/loss": loss.item(),
                "batch/accuracy": batch_acc,
                "batch/learning_rate": current_lr,
                "batch/batch_idx": batch_idx
            }
            
            if epoch is not None:
                log_data["batch/epoch"] = epoch
                # Use consistent step calculation: epoch * 1000 + batch_idx for unique batch steps
                step = epoch * 1000 + batch_idx
                wandb.log(log_data, step=step)
            else:
                # If no epoch provided, just use auto-increment (fallback)
                wandb.log(log_data)

    # Handle streaming dataloaders that might not have a proper length
    try:
        dataloader_len = len(train_loader)
        if dataloader_len == 0:
            dataloader_len = 1  # Avoid division by zero
    except (TypeError, ValueError):
        # For streaming dataloaders, use batch count
        dataloader_len = batch_idx + 1
    
    epoch_loss = running_loss / dataloader_len
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, device, test_loader, criterion, use_amp: bool = False, epoch: int = None, use_wandb: bool = False):
    """Evaluate the model on test set (optionally with AMP)."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_val = criterion(output, target).item()
            test_loss += loss_val
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        # Intentionally no per-batch logging; epoch summary is printed in main.py

    # Handle streaming dataloaders that might not have a proper length
    try:
        dataloader_len = len(test_loader)
        if dataloader_len == 0:
            dataloader_len = 1  # Avoid division by zero
    except (TypeError, ValueError):
        # For streaming dataloaders, use batch count
        dataloader_len = 1  # Default for test loader
    
    test_loss /= dataloader_len
    test_acc = 100. * correct / total
    
    # Log validation results to wandb
    if use_wandb and WANDB_AVAILABLE and epoch is not None:
        # Use epoch-based step for validation (consistent with main.py epoch logging)
        wandb.log({
            "val/epoch_loss": test_loss,
            "val/epoch_accuracy": test_acc,
            "val/epoch": epoch
        }, step=epoch)

    return test_loss, test_acc

