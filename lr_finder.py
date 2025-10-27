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
    
    # Get suggested LR using steepest descent point (better than minimum loss)
    losses = lr_finder.history['loss']
    lrs = lr_finder.history['lr']
    
    # Find steepest descent point (minimum gradient = steepest negative slope)
    if len(losses) > 1:
        gradients = []
        for i in range(1, len(losses)):
            grad = (losses[i] - losses[i-1]) / (lrs[i] - lrs[i-1])
            gradients.append(grad)
        # Find the point with steepest negative gradient
        steepest_idx = np.argmin(gradients) + 1
        suggested_lr = lrs[steepest_idx]
    else:
        # Fallback to minimum loss if only one point
        suggested_lr = lrs[losses.index(min(losses))]
    
    print(f"\nSuggested learning rate (steepest descent): {suggested_lr:.6f}")
    
    # Create and save plot if requested
    fig = None
    if plot and save_path:
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(os.path.abspath(save_path))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Use torch-lr-finder's built-in plot function
            fig = lr_finder.plot(skip_start=10, skip_end=5)
            
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
    
    return suggested_lr, fig
