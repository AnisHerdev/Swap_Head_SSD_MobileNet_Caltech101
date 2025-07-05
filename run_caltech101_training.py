#!/usr/bin/env python3
"""
Complete Caltech 101 training pipeline
This script downloads the dataset, prepares it, and trains the hybrid model
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Command completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with error: {e}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return False


def main():
    """Main function to run the complete training pipeline"""
    parser = argparse.ArgumentParser(description='Complete Caltech 101 training pipeline')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default='./caltech101',
                       help='Directory to save the original dataset')
    parser.add_argument('--processed_dir', type=str, default='./caltech101_processed',
                       help='Directory to save the processed dataset')
    parser.add_argument('--output_dir', type=str, default='./caltech101_output',
                       help='Output directory for training results')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip downloading if dataset already exists')
    parser.add_argument('--skip_preparation', action='store_true',
                       help='Skip dataset preparation if already done')
    
    args = parser.parse_args()
    
    print("CALTECH 101 COMPLETE TRAINING PIPELINE")
    print("=" * 60)
    print("This script will:")
    print("1. Download Caltech 101 dataset")
    print("2. Prepare dataset splits (train/val/test)")
    print("3. Train the hybrid model")
    print("4. Evaluate the model")
    print("=" * 60)
    
    # Step 1: Download dataset
    if not args.skip_download:
        download_cmd = f"python download_caltech101.py --data_dir {args.data_dir} --output_dir {args.processed_dir}"
        if not run_command(download_cmd, "Downloading Caltech 101 dataset"):
            print("Failed to download dataset. Exiting.")
            return
    else:
        print("Skipping dataset download (--skip_download flag)")
    
    # Step 2: Prepare dataset (if not already done)
    if not args.skip_preparation:
        if not os.path.exists(args.processed_dir) or len(os.listdir(args.processed_dir)) == 0:
            prepare_cmd = f"python download_caltech101.py --data_dir {args.data_dir} --output_dir {args.processed_dir} --skip_download"
            if not run_command(prepare_cmd, "Preparing dataset splits"):
                print("Failed to prepare dataset. Exiting.")
                return
        else:
            print("Dataset already prepared, skipping preparation step")
    else:
        print("Skipping dataset preparation (--skip_preparation flag)")
    
    # Step 3: Train model
    train_cmd = f"python train_caltech101.py --data_dir {args.processed_dir} --output_dir {args.output_dir} --epochs {args.epochs} --batch_size {args.batch_size} --learning_rate {args.learning_rate}"
    if not run_command(train_cmd, "Training hybrid model on Caltech 101"):
        print("Failed to train model. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
    print(f"Model checkpoint: {args.output_dir}/best_model.pth")
    print(f"Training summary: {args.output_dir}/training_summary.json")
    print(f"Class mapping: {args.output_dir}/class_mapping.json")
    
    # Instructions for inference
    print("\nTo run inference on new images:")
    print(f"python inference.py --model_path {args.output_dir}/best_model.pth --class_mapping_path {args.output_dir}/class_mapping.json --image_path /path/to/image.jpg")


if __name__ == "__main__":
    main() 