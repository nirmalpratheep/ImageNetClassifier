#!/usr/bin/env python3
"""
Test script for streaming ImageNet-1K dataset functionality.
This script tests the Hugging Face datasets integration with streaming.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def test_streaming_dataset():
    """Test the streaming dataset functionality."""
    
    print("Testing Streaming ImageNet-1K Dataset")
    print("="*50)
    
    # Check if datasets library is available
    try:
        from datasets import load_dataset
        print("✓ datasets library is available")
    except ImportError:
        print("✗ datasets library is not available")
        print("Please install with: pip install datasets")
        return False
    
    # Test loading datasets
    try:
        from preprocess import get_data_loaders, get_datasets
        
        print("\nTesting dataset loading...")
        
        # Test with streaming and limited samples
        print("Loading ImageNet-1K with streaming (limited to 100 samples)...")
        train_loader, test_loader = get_data_loaders(
            batch_size=16,
            dataset_name="imagenet1k",
            streaming=True,
            max_samples=100,
            num_workers=0,  # Use 0 workers for testing
            pin_memory=False
        )
        
        print("✓ Datasets loaded successfully")
        
        # Test data loading
        print("\nTesting data loading...")
        try:
            # Get a few batches
            batch_count = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                print(f"Batch {batch_idx}: data shape {data.shape}, labels shape {target.shape}")
                batch_count += 1
                if batch_count >= 3:  # Test first 3 batches
                    break
            
            print(f"✓ Successfully loaded {batch_count} batches")
            print(f"✓ Data shape: {data.shape}")
            print(f"✓ Labels shape: {target.shape}")
            print(f"✓ Data type: {data.dtype}")
            print(f"✓ Labels type: {target.dtype}")
            
        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            return False
        
        # Test with different batch sizes
        print("\nTesting different batch sizes...")
        for batch_size in [8, 16, 32]:
            try:
                train_loader_bs, _ = get_data_loaders(
                    batch_size=batch_size,
                    dataset_name="imagenet1k",
                    streaming=True,
                    max_samples=50,
                    num_workers=0
                )
                
                # Test one batch
                data, target = next(iter(train_loader_bs))
                print(f"✓ Batch size {batch_size}: data shape {data.shape}")
                
            except Exception as e:
                print(f"✗ Batch size {batch_size} failed: {e}")
                return False
        
        print("✓ All batch size tests passed")
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    
    return True

def test_lr_finder_with_streaming():
    """Test LR finder with streaming dataset."""
    
    print("\n" + "="*50)
    print("Testing LR Finder with Streaming Dataset")
    print("="*50)
    
    try:
        from preprocess import get_data_loaders
        from lr_finder import find_lr
        import torch.optim as optim
        
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
        
        print(f"Using device: {device}")
        
        # Load streaming dataset
        train_loader, _ = get_data_loaders(
            batch_size=16,
            dataset_name="imagenet1k",
            streaming=True,
            max_samples=200,  # Small number for testing
            num_workers=0
        )
        
        print("Running LR finder with streaming dataset...")
        
        # Run LR finder
        suggested_lr, fig = find_lr(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            start_lr=1e-6,
            end_lr=1,
            num_iter=20,  # Small number for testing
            plot=False,  # Don't plot in test
            save_path=None
        )
        
        print(f"✓ LR finder completed successfully!")
        print(f"✓ Suggested learning rate: {suggested_lr:.2e}")
        
        return True
        
    except Exception as e:
        print(f"✗ LR finder test failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("Streaming Dataset Test Suite")
    print("="*50)
    
    # Run tests
    streaming_test_passed = test_streaming_dataset()
    lr_finder_test_passed = test_lr_finder_with_streaming()
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Streaming Dataset Test: {'PASSED' if streaming_test_passed else 'FAILED'}")
    print(f"LR Finder with Streaming Test: {'PASSED' if lr_finder_test_passed else 'FAILED'}")
    
    if streaming_test_passed and lr_finder_test_passed:
        print("\n🎉 All tests passed! Streaming functionality is working correctly.")
        print("\nNext steps:")
        print("1. Run LR finder: python run_lr_finder.py --dataset imagenet1k --max_samples 1000")
        print("2. The dataset will be streamed from Hugging Face - no local download required")
        print("3. Check the generated LR finder plot")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    print("="*50)

if __name__ == "__main__":
    main()

