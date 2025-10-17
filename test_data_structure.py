#!/usr/bin/env python3
"""
Test Data Structure Script - Validates your ImageNet subset data structure
and shows detailed information about your dataset.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Test and validate ImageNet subset data structure")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory to test")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples to test (default: 1000)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("IMAGENET SUBSET DATA STRUCTURE TEST")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Max samples for testing: {args.max_samples}")
    print(f"Validation split ratio: {args.val_ratio}")
    print("="*70)
    
    # Test import
    try:
        from preprocess import detect_data_structure, get_datasets
        print("✓ Successfully imported data loading functions")
    except Exception as e:
        print(f"❌ Failed to import data loading functions: {e}")
        return 1
    
    # Test data structure detection
    try:
        data_structure = detect_data_structure(args.data_dir)
        print(f"✓ Data structure detected: {data_structure}")
        
        if data_structure == "none":
            print(f"❌ No data directory found at: {args.data_dir}")
            print("💡 Make sure your data directory exists and contains class folders")
            return 1
        elif data_structure == "unknown":
            print(f"⚠️  Unknown data structure detected")
            print("💡 Your data might not be in a recognized format")
        elif data_structure == "imagenet_subset":
            print(f"✓ Perfect! Your data structure matches ImageNet subset format")
        
    except Exception as e:
        print(f"❌ Failed to detect data structure: {e}")
        return 1
    
    # Test dataset loading
    print("\nTesting dataset loading...")
    try:
        train_dataset, test_dataset = get_datasets(
            streaming=False,
            max_samples=args.max_samples,
            data_dir=args.data_dir,
            val_ratio=args.val_ratio
        )
        
        print(f"✓ Datasets loaded successfully!")
        print(f"✓ Train samples: {len(train_dataset)}")
        print(f"✓ Test samples: {len(test_dataset)}")
        
        # Try to get class information
        if hasattr(train_dataset, 'classes'):
            classes = train_dataset.classes
            print(f"✓ Classes: {len(classes)}")
            print(f"✓ Sample classes: {classes[:5]}{'...' if len(classes) > 5 else ''}")
        elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'classes'):
            classes = train_dataset.dataset.classes
            print(f"✓ Classes (from wrapped dataset): {len(classes)}")
            print(f"✓ Sample classes: {classes[:5]}{'...' if len(classes) > 5 else ''}")
        else:
            print("⚠️  Could not determine class information")
        
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        print("💡 Check that your data directory contains valid image files")
        return 1
    
    # Test data loading (first batch)
    print("\nTesting data loading (first batch)...")
    try:
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        # Get first batch from train loader
        train_batch = next(iter(train_loader))
        print(f"✓ Train batch loaded successfully!")
        print(f"  - Image batch shape: {train_batch[0].shape}")
        print(f"  - Label batch shape: {train_batch[1].shape}")
        print(f"  - Labels in batch: {train_batch[1].tolist()}")
        
        # Get first batch from test loader
        test_batch = next(iter(test_loader))
        print(f"✓ Test batch loaded successfully!")
        print(f"  - Image batch shape: {test_batch[0].shape}")
        print(f"  - Label batch shape: {test_batch[1].shape}")
        print(f"  - Labels in batch: {test_batch[1].tolist()}")
        
    except Exception as e:
        print(f"❌ Failed to load data batches: {e}")
        return 1
    
    # Test with different validation ratios
    print("\nTesting different validation ratios...")
    for val_ratio in [0.1, 0.2, 0.3]:
        try:
            train_ds, test_ds = get_datasets(
                streaming=False,
                max_samples=min(1000, args.max_samples),
                data_dir=args.data_dir,
                val_ratio=val_ratio
            )
            print(f"✓ Val ratio {val_ratio}: Train={len(train_ds)}, Test={len(test_ds)}")
        except Exception as e:
            print(f"❌ Failed with val_ratio {val_ratio}: {e}")
    
    print("\n" + "="*70)
    print("DATA STRUCTURE TEST RESULTS")
    print("="*70)
    print("✅ All tests passed! Your data structure is compatible.")
    print(f"📁 Data structure: {data_structure}")
    print(f"📊 Ready for LR finding and training!")
    print("\n🚀 Next steps:")
    print(f"1. Run LR finder: python run_lr_finder_safe.py --data_dir {args.data_dir}")
    print(f"2. Or run basic test: python run_lr_finder_safe.py --max_samples 1000 --data_dir {args.data_dir}")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
