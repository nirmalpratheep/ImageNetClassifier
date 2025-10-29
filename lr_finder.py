import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent blocking
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
    step_mode: str = "exp",
    smooth_f: float = 0.05,
    diverge_th: float = 5,
    plot: bool = True,
    save_path: Optional[str] = None,
    use_amp: bool = False
) -> Tuple[float, Optional[plt.Figure]]:
    """
    Find optimal learning rate using torch-lr-finder library.
    
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
        use_amp: Use automatic mixed precision (not used in torch-lr-finder)
        
    Returns:
        Tuple of (suggested_lr, figure)
    """
    print("Running learning rate range test using torch-lr-finder...")
    
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
    
    # Get suggested LR from the plot method
    # Use the plot method with suggested_lr=True to get the optimal LR
    fig = None
    suggested_lr = None
    
    if plot:
        try:
            # Use torch-lr-finder's built-in plot function with suggested_lr=True
            # This will return the figure and may also set/return the suggested learning rate
            result = lr_finder.plot(skip_start=10, skip_end=10, suggested_lr=True)
            
            # Handle different return types from plot method
            if isinstance(result, tuple):
                fig = result[0]
                if len(result) > 1:
                    suggested_lr = float(result[1])
                    print(f"Suggested learning rate from plot: {suggested_lr:.6f}")
            else:
                fig = result
                
            # Also check if lr_finder has a suggested_lr attribute set by the plot method
            if suggested_lr is None and hasattr(lr_finder, 'suggested_lr'):
                suggested_lr = float(lr_finder.suggested_lr)
                print(f"Suggested learning rate from plot attribute: {suggested_lr:.6f}")
            
            # Save the plot if path provided
            if save_path:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(os.path.abspath(save_path))
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # Save the plot
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"LR finder plot saved to: {save_path}")
                
                # Close figure to prevent blocking
                plt.close(fig)
        
        except Exception as e:
            print(f"Warning: Could not create plot: {e}")
            fig = None
    
    # Reset model and optimizer to original state
    lr_finder.reset()
    
    # Ensure we return a Python float
    return_lr = float(suggested_lr) if suggested_lr else float(start_lr * 10)
    
    print(f"Final suggested learning rate: {return_lr:.6f}")
    
    return return_lr, fig
