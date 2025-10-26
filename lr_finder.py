import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
import os

# Import torch-lr-finder - this is required
from torch_lr_finder import LRFinder as TorchLRFinder


def _print_lr_summary_table(losses, lrs, suggested_lr, min_loss_idx, steepest_idx, save_path=None):
    """Print and save a detailed table of LR finder results."""
    
    # Calculate OneCycleLR parameters (assuming defaults)
    div_factor = 25.0
    final_div_factor = 10000.0
    
    initial_lr = suggested_lr / div_factor
    min_lr = initial_lr / final_div_factor
    
    # Print summary
    print("\n" + "="*70)
    print("LEARNING RATE FINDER - DETAILED SUMMARY")
    print("="*70)
    
    # Sample key iterations
    print("\nSample LR Range Test Results:")
    print(f"{'Iteration':<12} {'LR':<12} {'Loss':<12} {'Status':<25}")
    print("-"*65)
    
    # Show a sample of iterations
    sample_indices = [0, len(losses)//4, len(losses)//2, steepest_idx-1, steepest_idx, 
                     min(min(steepest_idx+1, len(losses)-1), steepest_idx+2), min_loss_idx, len(losses)-1]
    sample_indices = sorted(set([max(0, min(idx, len(losses)-1)) for idx in sample_indices]))
    
    for idx in sample_indices:
        lr = lrs[idx]
        loss = losses[idx]
        if idx == steepest_idx:
            status = "← STEEPEST DESCENT (SELECTED!)"
        elif idx == min_loss_idx:
            status = "← Minimum Loss"
        elif loss < losses[0] * 0.9 and idx > steepest_idx:
            status = "← Good zone (diverging soon)"
        elif loss < losses[0] * 0.9:
            status = "← Good zone"
        elif loss > losses[0] * 1.2:
            status = "← Diverging"
        else:
            status = "← Too small"
        print(f"{idx:<12} {lr:<12.2e} {loss:<12.4f} {status}")
    
    print("\n" + "="*70)
    print("SUGGESTED LEARNING RATE")
    print("="*70)
    print(f"Steepest Descent Point: {suggested_lr:.6f} ({suggested_lr:.2e})")
    print(f"Minimum Loss Point:     {lrs[min_loss_idx]:.6f} ({lrs[min_loss_idx]:.2e})")
    
    print("\n" + "="*70)
    print("ONECYCLELR SCHEDULER - LR RANGE CALCULATION")
    print("="*70)
    print(f"Suggested LR (max_lr):     {suggested_lr:.6f} (Used as OneCycleLR peak)")
    print(f"Initial LR:                {initial_lr:.6f} (max_lr / {div_factor})")
    print(f"Final LR (min_lr):         {min_lr:.8f} (initial_lr / {final_div_factor})")
    print(f"\nOneCycleLR Schedule:")
    print(f"  - Starts at:   {initial_lr:.6f}")
    print(f"  - Reaches max: {suggested_lr:.6f} (at ~30% through training)")
    print(f"  - Ends at:     {min_lr:.8f}")
    print(f"\nThis range will be used during actual training.")
    print("="*70 + "\n")
    
    # Save to file
    if save_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(save_path))
        if output_dir:  # Only create if save_path has a directory
            os.makedirs(output_dir, exist_ok=True)
        
        summary_file = save_path.replace('.png', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("LEARNING RATE FINDER - DETAILED SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Suggested LR: {suggested_lr:.6f}\n")
            f.write(f"Minimum Loss LR: {lrs[min_loss_idx]:.6f}\n")
            f.write(f"Initial LR (OneCycle): {initial_lr:.6f}\n")
            f.write(f"Final LR (OneCycle): {min_lr:.8f}\n")
            f.write(f"\nFull LR History:\n")
            f.write(f"{'Iteration':<12} {'LR':<15} {'Loss':<15}\n")
            f.write("-"*45 + "\n")
            for i, (lr, loss) in enumerate(zip(lrs, losses)):
                f.write(f"{i:<12} {lr:<15.6e} {loss:<15.4f}\n")
        print(f"Full summary saved to: {summary_file}\n")


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
    min_loss_idx = losses.index(min(losses))
    
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
        steepest_idx = min_loss_idx
        suggested_lr = lrs[min_loss_idx]
    
    print(f"Suggested learning rate: {suggested_lr:.2e}")
    
    # Print detailed summary table
    _print_lr_summary_table(losses, lrs, suggested_lr, min_loss_idx, steepest_idx, save_path)
    
    # Create plot if requested
    fig = None
    if plot:
        try:
            # Turn off interactive mode to prevent blocking
            plt.ioff()
            
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
            
            # Close figure to prevent display blocking
            plt.close(fig)
            
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
        steepest_idx = min_loss_idx
        steepest_lr = min_loss_lr
    
    # Use steepest descent as it's more commonly recommended
    suggested_lr = steepest_lr
    
    print(f"Minimum loss LR: {min_loss_lr:.2e}")
    print(f"Steepest descent LR: {steepest_lr:.2e}")
    print(f"Suggested learning rate: {suggested_lr:.2e}")
    
    # Print detailed summary table
    _print_lr_summary_table(losses, lrs, suggested_lr, min_loss_idx, steepest_idx, save_path)
    
    # Create plot if requested
    fig = None
    if plot:
        try:
            # Turn off interactive mode to prevent blocking
            plt.ioff()
            
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
            
            # Close figure to prevent display blocking
            plt.close(fig)
                
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
