#!/usr/bin/env python3
"""
Test script for LR finder functionality.
This script tests the LR finder implementation without requiring actual data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def test_lr_finder():
    """Test the LR finder functionality with dummy data."""
    
    print("Testing LR Finder Implementation...")
    print("="*50)
    
    # Check if torch-lr-finder is available (required)
    try:
        from torch_lr_finder import LRFinder
        print("[OK] torch-lr-finder is available")
    except ImportError:
        print("[ERROR] torch-lr-finder is not available")
        print("torch-lr-finder is REQUIRED for this implementation")
        print("Please install with: pip install torch-lr-finder")
        return False
    
    # Create dummy data
    print("\nCreating dummy dataset...")
    dummy_data = torch.randn(100, 3, 224, 224)  # ImageNet-like data
    dummy_labels = torch.randint(0, 1000, (100,))  # 1000 classes for ImageNet
    
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"[OK] Created dataset with {len(dataset)} samples")
    print(f"[OK] Data shape: {dummy_data.shape}")
    print(f"[OK] Labels shape: {dummy_labels.shape}")
    
    # Create a simple model
    print("\nCreating model...")
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 1000)  # 1000 classes for ImageNet
    )
    
    print("[OK] Created simple CNN model")
    
    # Create optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[OK] Using device: {device}")
    
    # Test our LR finder wrapper
    try:
        from lr_finder import find_lr
        
        print("\nTesting LR finder...")
        suggested_lr, fig = find_lr(
            model=model,
            train_loader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            start_lr=1e-6,
            end_lr=1,
            num_iter=20,  # Small number for testing
            plot=False,  # Don't plot in test
            save_path=None
        )
        
        print(f"[OK] LR finder completed successfully!")
        print(f"[OK] Suggested learning rate: {suggested_lr:.2e}")
        
    except Exception as e:
        print(f"[ERROR] LR finder test failed: {e}")
        return False
    
    # Test advanced LR finder
    try:
        from lr_finder import find_lr_advanced
        
        print("\nTesting advanced LR finder...")
        suggested_lr_adv, fig_adv = find_lr_advanced(
            model=model,
            train_loader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            start_lr=1e-6,
            end_lr=1,
            num_iter=20,
            plot=False,
            save_path=None
        )
        
        print(f"[OK] Advanced LR finder completed successfully!")
        print(f"[OK] Suggested learning rate: {suggested_lr_adv:.2e}")
        
    except Exception as e:
        print(f"[ERROR] Advanced LR finder test failed: {e}")
        return False
    
    print("\n" + "="*50)
    print("[OK] All tests passed successfully!")
    print("[OK] LR finder implementation is working correctly")
    
    return True

def test_dataset_support():
    """Test dataset support functionality."""
    
    print("\nTesting Dataset Support...")
    print("="*50)
    
    try:
        from preprocess import get_data_loaders, get_transforms
        
        # Test CIFAR-100 transforms
        print("Testing CIFAR-100 transforms...")
        train_transforms, test_transforms = get_transforms("cifar100")
        print("[OK] CIFAR-100 transforms created successfully")
        
        # Test ImageNet transforms
        print("Testing ImageNet transforms...")
        train_transforms, test_transforms = get_transforms("imagenet1k")
        print("[OK] ImageNet transforms created successfully")
        
        # Test model building
        from model_resnet50 import build_model
        
        print("Testing model building...")
        device = torch.device('cpu')  # Use CPU for testing
        
        # Test CIFAR model
        model_cifar = build_model(device, num_classes=100, input_size=32)
        print("[OK] CIFAR model built successfully")
        
        # Test ImageNet model
        model_imagenet = build_model(device, num_classes=1000, input_size=224)
        print("[OK] ImageNet model built successfully")
        
        print("[OK] Dataset support tests passed!")
        
    except Exception as e:
        print(f"[ERROR] Dataset support test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("LR Finder Test Suite")
    print("="*50)
    
    # Run tests
    lr_test_passed = test_lr_finder()
    dataset_test_passed = test_dataset_support()
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"LR Finder Test: {'PASSED' if lr_test_passed else 'FAILED'}")
    print(f"Dataset Support Test: {'PASSED' if dataset_test_passed else 'FAILED'}")
    
    if lr_test_passed and dataset_test_passed:
        print("\n[SUCCESS] All tests passed! The implementation is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run LR finder: python run_lr_finder.py --dataset imagenet1k")
        print("3. Check the generated plot and suggested learning rate")
    else:
        print("\n[FAILED] Some tests failed. Please check the error messages above.")
    
    print("="*50)
