#!/usr/bin/env python3
"""
Training script for Caltech 101 dataset using the hybrid model
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from models.hybrid_model import create_hybrid_model
from utils.data_utils import create_data_loaders, get_class_names, save_class_mapping, calculate_dataset_stats
from utils.training_utils import train_model, evaluate_model, load_trained_model


def setup_caltech101_training(data_dir, output_dir, batch_size=16, num_workers=4):
    """
    Setup training for Caltech 101 dataset
    """
    print("=" * 60)
    print("CALTECH 101 TRAINING SETUP")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory not found at {data_dir}")
        print("Please run download_caltech101.py first to download and prepare the dataset.")
        return None, None, None, None, None
    
    # Calculate dataset statistics
    print("Analyzing dataset...")
    stats = calculate_dataset_stats(data_dir)
    print(f"Dataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Number of classes: {stats['num_classes']}")
    print(f"  Classes: {stats['class_names'][:10]}...")  # Show first 10 classes
    
    # Get class names and number of classes
    class_names = get_class_names(data_dir)
    num_classes = len(class_names)
    
    print(f"\nTraining Configuration:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input size: 320x320")
    print(f"  Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save class mapping
    class_mapping_path = os.path.join(output_dir, 'class_mapping.json')
    save_class_mapping(data_dir, class_mapping_path)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        input_size=320
    )
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, num_classes, class_names


def create_caltech101_model(num_classes, pretrained=True):
    """
    Create model optimized for Caltech 101
    """
    print("Creating hybrid model for Caltech 101...")
    
    model = create_hybrid_model(num_classes=num_classes, pretrained=pretrained)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return model


def train_caltech101_model(model, train_loader, val_loader, output_dir, 
                          epochs=50, learning_rate=0.001, weight_decay=1e-4,
                          early_stopping_patience=15, device=None):
    """
    Train the model on Caltech 101 dataset
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    
    # Move model to device
    model = model.to(device)
    
    # Train the model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    training_info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        save_dir=output_dir,
        early_stopping_patience=early_stopping_patience
    )
    
    return training_info


def evaluate_caltech101_model(model, test_loader, class_names, output_dir, device=None):
    """
    Evaluate the trained model on Caltech 101 test set
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_dir=output_dir
    )
    
    return results


def main():
    """Main training function for Caltech 101"""
    parser = argparse.ArgumentParser(description='Train hybrid model on Caltech 101 dataset')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./caltech101_processed',
                       help='Path to the processed Caltech 101 dataset')
    parser.add_argument('--output_dir', type=str, default='./caltech101_output',
                       help='Output directory for checkpoints and results')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Model arguments
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup dataset and data loaders
    train_loader, val_loader, test_loader, num_classes, class_names = setup_caltech101_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if train_loader is None:
        return
    
    # Create model
    model = create_caltech101_model(
        num_classes=num_classes,
        pretrained=args.pretrained
    )
    
    # Train model
    training_info = train_caltech101_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        device=device
    )
    
    # Evaluate model
    results = evaluate_caltech101_model(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        output_dir=args.output_dir,
        device=device
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best validation accuracy: {training_info['best_val_acc']:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Test metrics: {results['metrics']}")
    print(f"Model saved to: {training_info['checkpoint_path']}")
    print(f"Results saved to: {args.output_dir}")
    
    # Save training summary
    summary = {
        'dataset': 'Caltech 101',
        'num_classes': num_classes,
        'best_val_acc': training_info['best_val_acc'],
        'test_accuracy': results['accuracy'],
        'test_metrics': results['metrics'],
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'early_stopping_patience': args.early_stopping_patience
        },
        'model_path': training_info['checkpoint_path'],
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {summary_path}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main() 