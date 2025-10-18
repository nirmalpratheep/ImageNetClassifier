#!/usr/bin/env python3
"""
Test matplotlib functionality on the current system
"""

import os
import sys

def test_matplotlib():
    print("="*50)
    print("MATPLOTLIB SYSTEM TEST")
    print("="*50)
    
    try:
        # Test 1: Basic import
        print("1. Testing matplotlib import...")
        import matplotlib
        print(f"   ✅ Matplotlib version: {matplotlib.__version__}")
        
        # Test 2: Backend configuration
        print("2. Testing backend configuration...")
        current_backend = matplotlib.get_backend()
        print(f"   Current backend: {current_backend}")
        
        # Set to Agg (non-GUI backend)
        matplotlib.use('Agg')
        new_backend = matplotlib.get_backend()
        print(f"   Backend after setting to Agg: {new_backend}")
        
        # Test 3: Pyplot import
        print("3. Testing pyplot import...")
        import matplotlib.pyplot as plt
        print("   ✅ Pyplot imported successfully")
        
        # Test 4: Create simple plot
        print("4. Testing plot creation...")
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 2, 3, 5]
        ax.plot(x, y, 'b-', marker='o')
        ax.set_title('Test Plot')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        print("   ✅ Plot created successfully")
        
        # Test 5: Save to file
        print("5. Testing file save...")
        test_dir = "./test_output"
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "matplotlib_test.png")
        
        print(f"   Attempting to save to: {test_file}")
        fig.savefig(test_file, dpi=150, bbox_inches='tight')
        
        # Verify file was created
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            print(f"   ✅ File saved successfully: {test_file} ({file_size} bytes)")
            return True
        else:
            print(f"   ❌ File was not created: {test_file}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        plt.close('all')

def test_torch_lr_finder_plot():
    print("\n" + "="*50)
    print("TORCH-LR-FINDER PLOT TEST")
    print("="*50)
    
    try:
        # Test if torch-lr-finder is available
        print("1. Testing torch-lr-finder import...")
        from torch_lr_finder import LRFinder
        print("   ✅ torch-lr-finder imported successfully")
        
        # This is just a basic import test - we can't easily create a mock LRFinder
        # without a full model and data loader setup
        print("   Note: Full LRFinder.plot() test requires model and data")
        return True
        
    except ImportError as e:
        print(f"   ❌ torch-lr-finder not available: {e}")
        print("   Install with: pip install torch-lr-finder")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("Testing matplotlib functionality for LR finder PNG generation...")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print()
    
    # Test basic matplotlib
    matplotlib_ok = test_matplotlib()
    
    # Test torch-lr-finder
    lr_finder_ok = test_torch_lr_finder_plot()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Matplotlib basic functionality: {'✅ OK' if matplotlib_ok else '❌ FAILED'}")
    print(f"torch-lr-finder import: {'✅ OK' if lr_finder_ok else '❌ FAILED'}")
    
    if matplotlib_ok and lr_finder_ok:
        print("\n🎉 All tests passed! Matplotlib should work for LR finder plots.")
        print("If PNG files are still not being created, the issue is likely in:")
        print("  - File path specification")
        print("  - Directory permissions")  
        print("  - LR finder plot() method returning None")
    else:
        print("\n⚠️  Some tests failed. This may explain why PNG files aren't being created.")
        
    # Check for any existing PNG files
    print(f"\nExisting PNG files in current directory:")
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    if png_files:
        for png in png_files:
            size = os.path.getsize(png)
            print(f"  - {png} ({size} bytes)")
    else:
        print("  - No PNG files found")

if __name__ == "__main__":
    main()
