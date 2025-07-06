#!/usr/bin/env python3
"""
Resume training script for the hybrid model (SSD320 + MobileNet)
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from models.hybrid_model import create_hybrid_model
from utils.data_utils import create_data_loaders, get_class_names, save_class_mapping
from utils.training_utils import resume_training, evaluate_model, load_trained_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Resume Training Hybrid Model (SSD320 + MobileNet)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the checkpoint file to resume from')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to the dataset directory')
    parser.add_argument('--input_size', type=int, default=320,
                       help='Input image size (default: 320)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading (default: 4)')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (auto-detected if not specified)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights (default: True)')
    
    # Training arguments
    parser.add_argument('--additional_epochs', type=int, default=20,
                       help='Number of additional epochs to train (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for checkpoints and results (default: ./output)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    return parser.parse_args()


def setup_device(args):
    """Setup device for training"""
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
    
    return device


def setup_data_loaders(args):
    """Setup data loaders"""
    print(f"Loading data from: {args.data_dir}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def setup_model(args, device):
    """Setup the hybrid model"""
    print("Creating hybrid model...")
    
    # Get number of classes
    if args.num_classes is None:
        class_names = get_class_names(args.data_dir)
        num_classes = len(class_names)
        print(f"Auto-detected {num_classes} classes: {class_names}")
    else:
        num_classes = args.num_classes
        print(f"Using {num_classes} classes")
    
    # Create model
    model = create_hybrid_model(
        num_classes=num_classes,
        pretrained=args.pretrained
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, num_classes


def main():
    """Main resume training function"""
    args = parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found: {args.checkpoint_path}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = setup_device(args)
    
    # Save class mapping
    save_class_mapping(args.data_dir, os.path.join(args.output_dir, 'class_mapping.json'))
    
    # Setup data loaders
    train_loader, val_loader, test_loader = setup_data_loaders(args)
    
    # Setup model
    model, num_classes = setup_model(args, device)
    
    # Resume training
    print("\n" + "="*60)
    print("RESUMING TRAINING")
    print("="*60)
    
    training_info = resume_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=args.checkpoint_path,
        num_epochs=args.additional_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=args.output_dir,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Evaluate model
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    # Get class names for evaluation
    class_names = get_class_names(args.data_dir)
    
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_dir=args.output_dir
    )
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best validation accuracy: {training_info['best_val_acc']:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Test metrics: {results['metrics']}")
    print(f"Model saved to: {training_info['checkpoint_path']}")
    print(f"Results saved to: {args.output_dir}")
    
    print("\nResume training completed successfully!")

    # Save best model (for best performance)
    if training_info['best_val_acc'] > training_info['best_val_acc']:
        torch.save({
            'epoch': training_info['epoch'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': training_info['optimizer'].state_dict(),
            'scheduler_state_dict': training_info['scheduler'].state_dict(),
            'val_acc': training_info['best_val_acc'],
            'metrics': training_info['best_metrics']
        }, os.path.join(args.output_dir, 'best_model.pth'))

    # Save last model (for resuming)
    torch.save({
        'epoch': training_info['epoch'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': training_info['optimizer'].state_dict(),
        'scheduler_state_dict': training_info['scheduler'].state_dict(),
        'val_acc': training_info['best_val_acc'],
        'train_acc': training_info['best_train_acc'],
        'metrics': training_info['best_metrics']
    }, os.path.join(args.output_dir, 'last_model.pth'))


if __name__ == "__main__":
    main() 