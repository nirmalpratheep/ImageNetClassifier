#!/usr/bin/env python3
"""
Isolated test for LR finder plot creation issue
"""

import os
import sys
import traceback

# Add the current directory to Python path to import our modules
sys.path.insert(0, os.getcwd())

def test_lr_finder_import():
    """Test if our lr_finder module can be imported correctly"""
    print("="*50)
    print("LR FINDER IMPORT TEST")
    print("="*50)
    
    try:
        print("1. Testing lr_finder module import...")
        from lr_finder import find_lr, find_lr_advanced
        print("   ✅ LR finder functions imported successfully")
        
        print("2. Testing torch-lr-finder import...")
        from torch_lr_finder import LRFinder
        print("   ✅ torch-lr-finder imported successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_minimal_lr_finder():
    """Test LR finder with minimal setup to isolate plot issue"""
    print("\n" + "="*50)
    print("MINIMAL LR FINDER PLOT TEST")
    print("="*50)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        print("1. Creating minimal model and data...")
        
        # Create tiny model
        model = nn.Linear(10, 5)
        print("   ✅ Model created")
        
        # Create tiny dataset
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=10)
        print("   ✅ Data loader created")
        
        # Create optimizer and criterion  
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        print("   ✅ Optimizer and criterion created")
        
        print("\n2. Testing LR finder with plot...")
        
        # Import our function
        from lr_finder import find_lr
        
        # Test with explicit debug
        print("   Calling find_lr with plot=True...")
        
        suggested_lr, fig = find_lr(
            model=model,
            train_loader=dataloader,
            optimizer=optimizer, 
            criterion=criterion,
            device=device,
            start_lr=1e-5,
            end_lr=1.0,
            num_iter=20,  # Very few iterations
            plot=True,
            save_path="./test_output/minimal_lr_test.png",
            use_amp=False,
            use_wandb=False
        )
        
        print(f"   ✅ LR finder completed! Suggested LR: {suggested_lr:.2e}")
        
        # Check if plot file was created
        plot_path = "./test_output/minimal_lr_test.png"
        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path)
            print(f"   ✅ Plot file created: {plot_path} ({file_size} bytes)")
            return True
        else:
            print(f"   ❌ Plot file NOT created: {plot_path}")
            return False
            
    except Exception as e:
        print(f"   ❌ LR finder test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("Testing LR finder plot creation in isolation...")
    
    # Create output directory
    os.makedirs("./test_output", exist_ok=True)
    
    # Test imports first
    import_ok = test_lr_finder_import()
    
    if not import_ok:
        print("\n❌ Can't proceed - import failures")
        return
        
    # Test minimal LR finder
    lr_finder_ok = test_minimal_lr_finder()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Import test: {'✅ OK' if import_ok else '❌ FAILED'}")
    print(f"LR finder plot test: {'✅ OK' if lr_finder_ok else '❌ FAILED'}")
    
    if lr_finder_ok:
        print("\n🎉 LR finder plot works! The issue is elsewhere.")
        print("Possible causes for main script issue:")
        print("  - Different data/model causing exceptions")
        print("  - Path/permission issues in main script")
        print("  - Exception handling in main.py")
    else:
        print("\n⚠️  LR finder plot is broken. This explains the PNG issue.")
        print("The debug output above shows the exact problem.")

if __name__ == "__main__":
    main()
