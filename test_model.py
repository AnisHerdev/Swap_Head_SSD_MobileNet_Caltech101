#!/usr/bin/env python3
"""
Test script for the hybrid model (SSD320 + MobileNet)
This script tests the model architecture, forward pass, and basic functionality.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import time

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from models.hybrid_model import create_hybrid_model, HybridModel, MobileNetFeatureExtractor, BridgingLayers, MobileNetClassifier


def test_model_creation():
    """Test model creation with different configurations"""
    print("Testing model creation...")
    
    # Test 1: Basic model creation
    try:
        model = create_hybrid_model(num_classes=10, pretrained=True)
        print("✓ Basic model creation successful")
    except Exception as e:
        print(f"✗ Basic model creation failed: {e}")
        return False
    
    # Test 2: Model with different number of classes
    try:
        model = create_hybrid_model(num_classes=5, pretrained=False)
        print("✓ Model with 5 classes creation successful")
    except Exception as e:
        print(f"✗ Model with 5 classes creation failed: {e}")
        return False
    
    # Test 3: Model without pretrained weights
    try:
        model = create_hybrid_model(num_classes=100, pretrained=False)
        print("✓ Model without pretrained weights creation successful")
    except Exception as e:
        print(f"✗ Model without pretrained weights creation failed: {e}")
        return False
    
    return True


def test_forward_pass():
    """Test forward pass with different input sizes"""
    print("\nTesting forward pass...")
    
    model = create_hybrid_model(num_classes=10, pretrained=False)
    model.eval()
    
    # Test different input sizes
    input_sizes = [224, 320, 416]
    
    for input_size in input_sizes:
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, input_size, input_size)
            
            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"✓ Forward pass successful with input size {input_size}x{input_size}")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected output shape: (1, 10)")
            
            # Check output shape
            if output.shape == (1, 10):
                print(f"  ✓ Output shape correct")
            else:
                print(f"  ✗ Output shape incorrect: expected (1, 10), got {output.shape}")
                return False
                
        except Exception as e:
            print(f"✗ Forward pass failed with input size {input_size}x{input_size}: {e}")
            return False
    
    return True


def test_model_components():
    """Test individual model components"""
    print("\nTesting model components...")
    
    # Test 1: Feature extractor
    try:
        feature_extractor = MobileNetFeatureExtractor(pretrained=False)
        dummy_input = torch.randn(1, 3, 320, 320)
        
        with torch.no_grad():
            features = feature_extractor(dummy_input)
        
        print("✓ Feature extractor test successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output features keys: {list(features.keys())}")
        
        # Check if layer4 exists (used by bridging layers)
        if 'layer4' in features:
            print(f"  ✓ layer4 feature shape: {features['layer4'].shape}")
        else:
            print(f"  ✗ layer4 feature not found")
            return False
            
    except Exception as e:
        print(f"✗ Feature extractor test failed: {e}")
        return False
    
    # Test 2: Bridging layers
    try:
        bridging_layers = BridgingLayers(input_channels=2048, output_size=1280)
        
        # Create dummy features
        dummy_features = {
            'layer4': torch.randn(1, 2048, 10, 10)
        }
        
        with torch.no_grad():
            adapted_features = bridging_layers(dummy_features)
        
        print("✓ Bridging layers test successful")
        print(f"  Input feature shape: {dummy_features['layer4'].shape}")
        print(f"  Output feature shape: {adapted_features.shape}")
        print(f"  Expected output shape: (1, 1280)")
        
        if adapted_features.shape == (1, 1280):
            print(f"  ✓ Output shape correct")
        else:
            print(f"  ✗ Output shape incorrect")
            return False
            
    except Exception as e:
        print(f"✗ Bridging layers test failed: {e}")
        return False
    
    # Test 3: Classifier
    try:
        classifier = MobileNetClassifier(num_classes=10, dropout=0.2)
        dummy_input = torch.randn(1, 1280)
        
        with torch.no_grad():
            output = classifier(dummy_input)
        
        print("✓ Classifier test successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: (1, 10)")
        
        if output.shape == (1, 10):
            print(f"  ✓ Output shape correct")
        else:
            print(f"  ✗ Output shape incorrect")
            return False
            
    except Exception as e:
        print(f"✗ Classifier test failed: {e}")
        return False
    
    return True


def test_model_parameters():
    """Test model parameter counting and freezing/unfreezing"""
    print("\nTesting model parameters...")
    
    model = create_hybrid_model(num_classes=10, pretrained=False)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if total_params > 0:
        print("✓ Parameter counting successful")
    else:
        print("✗ Parameter counting failed")
        return False
    
    # Test freezing feature extractor
    try:
        model.freeze_feature_extractor()
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Parameters after freezing feature extractor:")
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable parameters: {trainable_after_freeze:,}")
        
        if frozen_params > 0 and trainable_after_freeze < trainable_params:
            print("✓ Feature extractor freezing successful")
        else:
            print("✗ Feature extractor freezing failed")
            return False
            
    except Exception as e:
        print(f"✗ Feature extractor freezing failed: {e}")
        return False
    
    # Test unfreezing feature extractor
    try:
        model.unfreeze_feature_extractor()
        trainable_after_unfreeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Trainable parameters after unfreezing: {trainable_after_unfreeze:,}")
        
        if trainable_after_unfreeze == trainable_params:
            print("✓ Feature extractor unfreezing successful")
        else:
            print("✗ Feature extractor unfreezing failed")
            return False
            
    except Exception as e:
        print(f"✗ Feature extractor unfreezing failed: {e}")
        return False
    
    return True


def test_inference_speed():
    """Test inference speed with different batch sizes"""
    print("\nTesting inference speed...")
    
    model = create_hybrid_model(num_classes=10, pretrained=False)
    model.eval()
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16]
    input_size = 320
    
    for batch_size in batch_sizes:
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, input_size, input_size)
            
            # Warm up
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Time inference
            start_time = time.time()
            num_runs = 10
            
            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(dummy_input)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            
            print(f"✓ Batch size {batch_size}: {avg_time:.4f}s per inference")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            
        except Exception as e:
            print(f"✗ Inference speed test failed for batch size {batch_size}: {e}")
            return False
    
    return True


def test_memory_usage():
    """Test memory usage with different input sizes"""
    print("\nTesting memory usage...")
    
    model = create_hybrid_model(num_classes=10, pretrained=False)
    model.eval()
    
    # Test different input sizes
    input_sizes = [224, 320, 416]
    batch_size = 1
    
    for input_size in input_sizes:
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, input_size, input_size)
            
            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)
            
            # Calculate approximate memory usage
            input_memory = dummy_input.numel() * 4 / 1024 / 1024  # MB
            output_memory = output.numel() * 4 / 1024 / 1024  # MB
            
            print(f"✓ Input size {input_size}x{input_size}:")
            print(f"  Input memory: {input_memory:.2f} MB")
            print(f"  Output memory: {output_memory:.2f} MB")
            print(f"  Total I/O memory: {input_memory + output_memory:.2f} MB")
            
        except Exception as e:
            print(f"✗ Memory usage test failed for input size {input_size}: {e}")
            return False
    
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("HYBRID MODEL TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Model Components", test_model_components),
        ("Model Parameters", test_model_parameters),
        ("Inference Speed", test_inference_speed),
        ("Memory Usage", test_memory_usage)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed_tests += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 