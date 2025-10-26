#!/usr/bin/env python3
"""
Test script to verify ResNet50 initialization is correct.
"""

import torch
import torch.nn as nn
from model_resnet50 import ResNet50

def test_bn_initialization():
    """Test that BatchNorm layers are initialized correctly."""
    print("=" * 70)
    print("Testing ResNet50 BatchNorm Initialization")
    print("=" * 70)
    
    model = ResNet50(num_classes=1000)
    
    # Check bn3 weights (should be 0 in residual path)
    bn3_zeros = 0
    bn3_nonzeros = 0
    
    # Check downsample BN weights (should be 1, NOT 0!)
    downsample_bn_zeros = 0
    downsample_bn_ones = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if 'bn3' in name:
                if torch.allclose(module.weight, torch.zeros_like(module.weight)):
                    bn3_zeros += 1
                else:
                    bn3_nonzeros += 1
            elif 'downsample' in name:
                if torch.allclose(module.weight, torch.zeros_like(module.weight)):
                    downsample_bn_zeros += 1
                    print(f"❌ ERROR: {name} has γ=0 (should be 1!)")
                elif torch.allclose(module.weight, torch.ones_like(module.weight)):
                    downsample_bn_ones += 1
                    print(f"✓ {name} has γ=1 (correct!)")
    
    print()
    print("Results:")
    print("-" * 70)
    print(f"bn3 layers with γ=0 (residual path): {bn3_zeros} ✓")
    print(f"bn3 layers with γ≠0: {bn3_nonzeros}")
    print(f"downsample BN layers with γ=0: {downsample_bn_zeros} {'❌' if downsample_bn_zeros > 0 else '✓'}")
    print(f"downsample BN layers with γ=1: {downsample_bn_ones} ✓")
    print()
    
    if downsample_bn_zeros > 0:
        print("❌ FAIL: Downsample BN layers have γ=0 (blocks gradient flow!)")
        return False
    else:
        print("✓ PASS: All downsample BN layers have γ=1 (gradients can flow!)")
        return True

def test_forward_pass():
    """Test a forward pass to ensure model works."""
    print("=" * 70)
    print("Testing Forward Pass")
    print("=" * 70)
    
    model = ResNet50(num_classes=1000)
    model.eval()
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Output mean: {output.mean():.4f}")
    print(f"Output std: {output.std():.4f}")
    print()
    
    # Check if output is reasonable
    if output.shape == (batch_size, 1000):
        print("✓ PASS: Forward pass produces correct output shape")
        return True
    else:
        print("❌ FAIL: Forward pass produces wrong output shape")
        return False

def test_backward_pass():
    """Test backward pass to ensure gradients flow."""
    print("=" * 70)
    print("Testing Backward Pass (Gradient Flow)")
    print("=" * 70)
    
    model = ResNet50(num_classes=1000)
    model.train()
    
    # Create dummy input and target
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    target = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients are non-zero
    zero_grad_count = 0
    nonzero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.allclose(param.grad, torch.zeros_like(param.grad), atol=1e-10):
                zero_grad_count += 1
                if 'downsample' in name or 'layer1.0' in name:
                    print(f"⚠️  {name} has zero gradients")
            else:
                nonzero_grad_count += 1
    
    print()
    print(f"Parameters with non-zero gradients: {nonzero_grad_count}")
    print(f"Parameters with zero gradients: {zero_grad_count}")
    print(f"Loss value: {loss.item():.4f}")
    print()
    
    if zero_grad_count > 0:
        print(f"⚠️  WARNING: {zero_grad_count} parameters have zero gradients")
        print("   This might indicate gradient flow issues")
    
    if loss.item() > 0 and nonzero_grad_count > 0:
        print("✓ PASS: Backward pass produces gradients")
        return True
    else:
        print("❌ FAIL: Backward pass has issues")
        return False

if __name__ == "__main__":
    results = []
    
    results.append(test_bn_initialization())
    print()
    
    results.append(test_forward_pass())
    print()
    
    results.append(test_backward_pass())
    print()
    
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    if all(results):
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Please review the output above")
    print("=" * 70)

