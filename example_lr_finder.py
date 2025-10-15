#!/usr/bin/env python3
"""
Example script showing how to use the LR finder with torch-lr-finder.
This demonstrates the simplified usage with only torch-lr-finder library.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def main():
    print("LR Finder Example - Using torch-lr-finder only")
    print("="*50)
    
    # Check if torch-lr-finder is available
    try:
        from lr_finder import find_lr, find_lr_advanced
        print("✓ LR finder functions imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("Please install torch-lr-finder: pip install torch-lr-finder")
        return
    
    # Create dummy data (ImageNet-like)
    print("\nCreating dummy dataset...")
    dummy_data = torch.randn(200, 3, 224, 224)
    dummy_labels = torch.randint(0, 1000, (200,))
    
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 1000)
    )
    
    # Create optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"✓ Using device: {device}")
    
    # Example 1: Simple LR finder
    print("\n" + "="*50)
    print("EXAMPLE 1: Simple LR Finder")
    print("="*50)
    
    try:
        suggested_lr, fig = find_lr(
            model=model,
            train_loader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            start_lr=1e-6,
            end_lr=1,
            num_iter=50,
            plot=True,
            save_path="simple_lr_example.png"
        )
        
        print(f"✓ Simple LR finder completed!")
        print(f"✓ Suggested LR: {suggested_lr:.2e}")
        print(f"✓ Plot saved to: simple_lr_example.png")
        
    except Exception as e:
        print(f"✗ Simple LR finder failed: {e}")
    
    # Example 2: Advanced LR finder
    print("\n" + "="*50)
    print("EXAMPLE 2: Advanced LR Finder")
    print("="*50)
    
    try:
        suggested_lr_adv, fig_adv = find_lr_advanced(
            model=model,
            train_loader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            start_lr=1e-6,
            end_lr=1,
            num_iter=50,
            step_mode="exp",
            smooth_f=0.1,
            diverge_th=3,
            plot=True,
            save_path="advanced_lr_example.png"
        )
        
        print(f"✓ Advanced LR finder completed!")
        print(f"✓ Suggested LR: {suggested_lr_adv:.2e}")
        print(f"✓ Plot saved to: advanced_lr_example.png")
        
    except Exception as e:
        print(f"✗ Advanced LR finder failed: {e}")
    
    # Example 3: Direct torch-lr-finder usage
    print("\n" + "="*50)
    print("EXAMPLE 3: Direct torch-lr-finder Usage")
    print("="*50)
    
    try:
        from torch_lr_finder import LRFinder
        
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(dataloader, start_lr=1e-6, end_lr=1, num_iter=50)
        
        # Get suggested LR
        losses = lr_finder.history['loss']
        lrs = lr_finder.history['lr']
        suggested_lr_direct = lrs[losses.index(min(losses))]
        
        print(f"✓ Direct torch-lr-finder completed!")
        print(f"✓ Suggested LR: {suggested_lr_direct:.2e}")
        
        # Plot
        fig_direct = lr_finder.plot()
        fig_direct.savefig("direct_lr_example.png", dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: direct_lr_example.png")
        
        # Reset
        lr_finder.reset()
        
    except Exception as e:
        print(f"✗ Direct torch-lr-finder failed: {e}")
    
    print("\n" + "="*50)
    print("EXAMPLE COMPLETED")
    print("="*50)
    print("✓ All examples demonstrate torch-lr-finder usage")
    print("✓ No custom implementation - only torch-lr-finder library")
    print("✓ Check the generated PNG files for LR finder plots")

if __name__ == "__main__":
    main()

