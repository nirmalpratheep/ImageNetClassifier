import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
import os

# Import torch-lr-finder - this is required
from torch_lr_finder import LRFinder as TorchLRFinder


def find_lr(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 10,
    num_iter: int = 100,
    plot: bool = True,
    save_path: Optional[str] = None,
    use_amp: bool = False
) -> Tuple[float, Optional[plt.Figure]]:
    """
    Find optimal learning rate using torch-lr-finder library.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader (can be streaming)
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iter: Number of iterations
        plot: Whether to create a plot
        save_path: Path to save the plot
        use_amp: Use automatic mixed precision (not used in torch-lr-finder)
        
    Returns:
        Tuple of (suggested_lr, figure)
    """
    print("Running learning rate range test using torch-lr-finder...")
    
    # Handle streaming dataloader by creating a regular dataloader
    if hasattr(train_loader, '__class__') and 'StreamingDataLoader' in str(train_loader.__class__):
        print("Converting streaming dataloader to regular dataloader for LR finder...")
        # Create a regular dataloader from the streaming one
        from torch.utils.data import DataLoader, TensorDataset
        
        # Collect a few batches for LR finder
        batch_data = []
        batch_labels = []
        batch_count = 0
        max_batches = (num_iter + train_loader.batch_size - 1) // train_loader.batch_size
        
        for data, labels in train_loader:
            batch_data.append(data)
            batch_labels.append(labels)
            batch_count += 1
            if batch_count >= max_batches:
                break
        
        if batch_data:
            # Concatenate all batches
            all_data = torch.cat(batch_data, dim=0)
            all_labels = torch.cat(batch_labels, dim=0)
            
            # Create a regular dataset and dataloader
            lr_dataset = TensorDataset(all_data, all_labels)
            lr_dataloader = DataLoader(
                lr_dataset, 
                batch_size=train_loader.batch_size, 
                shuffle=True,
                num_workers=0,  # Use 0 workers for compatibility
                pin_memory=False
            )
        else:
            raise ValueError("No data collected from streaming dataloader")
    else:
        lr_dataloader = train_loader
    
    # Create LR finder
    lr_finder = TorchLRFinder(model, optimizer, criterion, device=device)
    
    # Run range test
    lr_finder.range_test(
        train_loader=lr_dataloader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter
    )
    
    # Get suggested LR using steepest descent point (more commonly used)
    losses = lr_finder.history['loss']
    lrs = lr_finder.history['lr']
    
    # Find steepest descent point (minimum gradient)
    if len(losses) > 1:
        # Calculate gradients (approximate)
        gradients = []
        for i in range(1, len(losses)):
            grad = (losses[i] - losses[i-1]) / (lrs[i] - lrs[i-1])
            gradients.append(grad)
        
        # Find the point with steepest negative gradient
        steepest_idx = np.argmin(gradients) + 1  # +1 because we started from index 1
        suggested_lr = lrs[steepest_idx]
    else:
        # Fallback to minimum loss if only one point
        suggested_lr = lrs[losses.index(min(losses))]
    
    print(f"Suggested learning rate: {suggested_lr:.2e}")
    
    # Create plot if requested
    fig = None
    if plot:
        try:
            # Use the built-in plot method from torch-lr-finder
            fig = lr_finder.plot(skip_start=10, skip_end=5)
            
            # Add suggested LR line
            ax = fig.gca()
            ax.axvline(x=suggested_lr, color='red', linestyle='--', alpha=0.7, 
                      label=f'Suggested LR: {suggested_lr:.2e}')
            ax.legend()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"LR finder plot saved to: {save_path}")
                
        except Exception as e:
            print(f"Warning: Could not create plot: {e}")
            fig = None
    
    # Reset model and optimizer to original state
    lr_finder.reset()
    
    return suggested_lr, fig


def find_lr_advanced(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 10,
    num_iter: int = 100,
    step_mode: str = "exp",
    smooth_f: float = 0.05,
    diverge_th: float = 5,
    plot: bool = True,
    save_path: Optional[str] = None
) -> Tuple[float, Optional[plt.Figure]]:
    """
    Advanced LR finder with more options using torch-lr-finder.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iter: Number of iterations
        step_mode: 'exp' for exponential, 'linear' for linear
        smooth_f: Smoothing factor for loss
        diverge_th: Threshold for divergence detection
        plot: Whether to create a plot
        save_path: Path to save the plot
        
    Returns:
        Tuple of (suggested_lr, figure)
    """
    print("Running advanced learning rate range test using torch-lr-finder...")
    
    # Create LR finder
    lr_finder = TorchLRFinder(model, optimizer, criterion, device=device)
    
    # Run range test with advanced options
    lr_finder.range_test(
        train_loader=train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode=step_mode,
        smooth_f=smooth_f,
        diverge_th=diverge_th
    )
    
    # Get suggested LR using different methods
    losses = lr_finder.history['loss']
    lrs = lr_finder.history['lr']
    
    # Method 1: Minimum loss point
    min_loss_idx = losses.index(min(losses))
    min_loss_lr = lrs[min_loss_idx]
    
    # Method 2: Steepest descent point
    if len(losses) > 1:
        gradients = []
        for i in range(1, len(losses)):
            grad = (losses[i] - losses[i-1]) / (lrs[i] - lrs[i-1])
            gradients.append(grad)
        
        steepest_idx = np.argmin(gradients) + 1
        steepest_lr = lrs[steepest_idx]
    else:
        steepest_lr = min_loss_lr
    
    # Use steepest descent as it's more commonly recommended
    suggested_lr = steepest_lr
    
    print(f"Minimum loss LR: {min_loss_lr:.2e}")
    print(f"Steepest descent LR: {steepest_lr:.2e}")
    print(f"Suggested learning rate: {suggested_lr:.2e}")
    
    # Create plot if requested
    fig = None
    if plot:
        try:
            fig = lr_finder.plot(skip_start=10, skip_end=5)
            
            # Add suggested LR lines
            ax = fig.gca()
            ax.axvline(x=min_loss_lr, color='blue', linestyle=':', alpha=0.7, 
                      label=f'Min Loss LR: {min_loss_lr:.2e}')
            ax.axvline(x=steepest_lr, color='red', linestyle='--', alpha=0.7, 
                      label=f'Steepest LR: {steepest_lr:.2e}')
            ax.legend()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"LR finder plot saved to: {save_path}")
                
        except Exception as e:
            print(f"Warning: Could not create plot: {e}")
            fig = None
    
    # Reset model and optimizer to original state
    lr_finder.reset()
    
    return suggested_lr, fig


# Backward compatibility - keep the old class name
class LRFinder:
    """
    Wrapper for torch-lr-finder LRFinder for backward compatibility.
    This is now just a direct wrapper around torch-lr-finder.
    """
    
    def __init__(self, model, optimizer, criterion, device, **kwargs):
        self.lr_finder = TorchLRFinder(model, optimizer, criterion, device=device, **kwargs)
    
    def range_test(self, train_loader, **kwargs):
        return self.lr_finder.range_test(train_loader, **kwargs)
    
    def plot(self, **kwargs):
        return self.lr_finder.plot(**kwargs)
    
    def reset(self):
        return self.lr_finder.reset()
    
    @property
    def history(self):
        return self.lr_finder.history
