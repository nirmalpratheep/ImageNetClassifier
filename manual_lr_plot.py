#!/usr/bin/env python3
"""
Manual LR finder plot creation - backup if torch-lr-finder.plot() fails
"""

import os
import matplotlib
matplotlib.use('Agg')  # Force headless backend
import matplotlib.pyplot as plt
import numpy as np

def create_manual_lr_plot(lrs, losses, suggested_lr=None, save_path="manual_lr_plot.png"):
    """
    Create LR finder plot manually from LR and loss data
    
    Args:
        lrs: List of learning rates
        losses: List of corresponding losses
        suggested_lr: Optional suggested learning rate to highlight
        save_path: Where to save the plot
    """
    
    print(f"🔍 Creating manual LR plot with {len(lrs)} data points")
    
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot LR vs Loss
        ax.plot(lrs, losses, 'b-', linewidth=2, label='Loss')
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder')
        ax.grid(True, alpha=0.3)
        
        # Add suggested LR line if provided
        if suggested_lr is not None:
            ax.axvline(x=suggested_lr, color='red', linestyle='--', alpha=0.7,
                      label=f'Suggested LR: {suggested_lr:.2e}')
            ax.legend()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Verify file creation
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"✅ Manual LR plot saved to: {save_path} ({file_size} bytes)")
            return True
        else:
            print(f"❌ Failed to save manual plot to: {save_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating manual plot: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        plt.close()

if __name__ == "__main__":
    # Test with dummy data
    print("Testing manual LR plot creation...")
    
    # Generate test data
    lrs = np.logspace(-6, 0, 100)  # 1e-6 to 1e0
    losses = 2.5 + np.random.normal(0, 0.1, 100) + 0.5 * np.log10(lrs)
    suggested_lr = 1e-3
    
    # Create test plot
    success = create_manual_lr_plot(lrs, losses, suggested_lr, "./test_output/manual_test.png")
    
    if success:
        print("🎉 Manual plot creation works!")
    else:
        print("❌ Manual plot creation failed")
