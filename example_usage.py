#!/usr/bin/env python3
"""
Example usage script for the hybrid model (SSD320 + MobileNet)
This script demonstrates how to use the hybrid model for training and inference.
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
import json

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from models.hybrid_model import create_hybrid_model
from utils.data_utils import create_data_loaders, get_class_names, save_class_mapping
from utils.training_utils import train_model, evaluate_model, load_trained_model
from inference import HybridModelInference


def example_1_basic_training():
    """
    Example 1: Basic training setup
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Training Setup")
    print("=" * 60)
    
    # Configuration
    data_dir = "path/to/your/dataset"  # Replace with your dataset path
    output_dir = "./example_output"
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get class information
    class_names = get_class_names(data_dir)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Save class mapping
    save_class_mapping(data_dir, os.path.join(output_dir, 'class_mapping.json'))
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        input_size=320
    )
    
    # Create model
    model = create_hybrid_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Train model
    training_info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir=output_dir
    )
    
    print(f"Training completed! Best validation accuracy: {training_info['best_val_acc']:.4f}")


def example_2_inference():
    """
    Example 2: Inference with trained model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Inference with Trained Model")
    print("=" * 60)
    
    # Configuration
    model_path = "./example_output/best_model.pth"
    class_mapping_path = "./example_output/class_mapping.json"
    image_path = "path/to/test/image.jpg"  # Replace with your test image
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run example_1_basic_training() first.")
        return
    
    # Initialize inference
    inference = HybridModelInference(
        model_path=model_path,
        class_mapping_path=class_mapping_path,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        input_size=320
    )
    
    # Make prediction
    if os.path.exists(image_path):
        predictions = inference.predict(image_path, top_k=3)
        
        print(f"Predictions for {image_path}:")
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['class_name']}: {pred['probability']:.4f} ({pred['probability']*100:.2f}%)")
    else:
        print(f"Test image not found at {image_path}")


def example_3_custom_model_configuration():
    """
    Example 3: Custom model configuration
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Model Configuration")
    print("=" * 60)
    
    # Create model with custom configuration
    num_classes = 5
    model = create_hybrid_model(num_classes=num_classes, pretrained=True)
    
    # Freeze feature extractor for transfer learning
    model.freeze_feature_extractor()
    print("Feature extractor frozen for transfer learning")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Unfreeze for fine-tuning
    model.unfreeze_feature_extractor()
    print("Feature extractor unfrozen for fine-tuning")


def example_4_batch_inference():
    """
    Example 4: Batch inference on multiple images
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Inference")
    print("=" * 60)
    
    # Configuration
    model_path = "./example_output/best_model.pth"
    class_mapping_path = "./example_output/class_mapping.json"
    image_dir = "path/to/test/images/"  # Replace with your test images directory
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run example_1_basic_training() first.")
        return
    
    # Initialize inference
    inference = HybridModelInference(
        model_path=model_path,
        class_mapping_path=class_mapping_path,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        input_size=320
    )
    
    # Get image paths
    if os.path.exists(image_dir):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(image_dir, filename))
        
        print(f"Found {len(image_paths)} images for batch inference")
        
        # Make predictions
        all_predictions = []
        for image_path in image_paths[:5]:  # Process first 5 images
            predictions = inference.predict(image_path, top_k=3)
            all_predictions.append({
                'image_path': image_path,
                'predictions': predictions
            })
            
            print(f"\nPredictions for {os.path.basename(image_path)}:")
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. {pred['class_name']}: {pred['probability']:.4f}")
        
        # Save predictions
        output_file = "./example_output/batch_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"\nBatch predictions saved to: {output_file}")
    else:
        print(f"Test directory not found at {image_dir}")


def example_5_model_analysis():
    """
    Example 5: Model analysis and debugging
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Model Analysis")
    print("=" * 60)
    
    # Create model
    model = create_hybrid_model(num_classes=10, pretrained=True)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 320, 320)
    
    # Test forward pass
    print("Testing forward pass...")
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output classes: {output.shape[1]}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    # Analyze model components
    print("\nModel component analysis:")
    
    # Feature extractor
    feature_extractor = model.feature_extractor
    print(f"  Feature extractor: {type(feature_extractor).__name__}")
    
    # Bridging layers
    bridging_layers = model.bridging_layers
    print(f"  Bridging layers: {type(bridging_layers).__name__}")
    
    # Classifier
    classifier = model.classifier
    print(f"  Classifier: {type(classifier).__name__}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


def main():
    """
    Run all examples
    """
    print("Hybrid Model (SSD320 + MobileNet) - Example Usage")
    print("=" * 60)
    
    # Example 1: Basic training (commented out as it requires dataset)
    # example_1_basic_training()
    print("Example 1: Basic Training (requires dataset - skipped)")
    
    # Example 2: Inference
    example_2_inference()
    
    # Example 3: Custom model configuration
    example_3_custom_model_configuration()
    
    # Example 4: Batch inference
    example_4_batch_inference()
    
    # Example 5: Model analysis
    example_5_model_analysis()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nTo run training example:")
    print("1. Prepare your dataset in the required format")
    print("2. Update the data_dir path in example_1_basic_training()")
    print("3. Uncomment the function call in main()")
    print("4. Run: python example_usage.py")


if __name__ == "__main__":
    main() 