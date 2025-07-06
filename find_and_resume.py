#!/usr/bin/env python3
"""
Helper script to find existing checkpoints and resume training
"""

import os
import sys
import argparse
from pathlib import Path
import glob

def find_checkpoints():
    """Find all checkpoint files in the current directory and subdirectories"""
    checkpoints = []
    
    # Look for common checkpoint patterns
    patterns = [
        "*.pth",
        "*.pt", 
        "best_model.pth",
        "checkpoint.pth",
        "model.pth"
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        checkpoints.extend(files)
    
    # Remove duplicates and sort
    checkpoints = list(set(checkpoints))
    checkpoints.sort()
    
    return checkpoints

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Find and resume training from checkpoints')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for checkpoints and results')
    parser.add_argument('--additional_epochs', type=int, default=20,
                       help='Number of additional epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--force_fresh', action='store_true',
                       help='Force fresh training even if checkpoints exist')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CHECKPOINT FINDER")
    print("="*60)
    
    # Find existing checkpoints
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("No checkpoint files found.")
        print("Starting fresh training...")
        
        # Start fresh training
        cmd = f"python train.py --data_dir {args.data_dir} --output_dir {args.output_dir} --epochs {args.additional_epochs} --learning_rate {args.learning_rate} --batch_size {args.batch_size}"
        print(f"\nRunning: {cmd}")
        os.system(cmd)
        
    else:
        print(f"Found {len(checkpoints)} checkpoint file(s):")
        for i, checkpoint in enumerate(checkpoints, 1):
            print(f"  {i}. {checkpoint}")
        
        if args.force_fresh:
            print("\nForce fresh training selected. Starting new training...")
            cmd = f"python train.py --data_dir {args.data_dir} --output_dir {args.output_dir} --epochs {args.additional_epochs} --learning_rate {args.learning_rate} --batch_size {args.batch_size}"
            print(f"\nRunning: {cmd}")
            os.system(cmd)
        else:
            print("\nOptions:")
            print("1. Resume from the most recent checkpoint")
            print("2. Choose a specific checkpoint")
            print("3. Start fresh training")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                # Resume from most recent checkpoint
                latest_checkpoint = checkpoints[-1]
                print(f"Resuming from: {latest_checkpoint}")
                cmd = f"python resume_training.py --checkpoint_path {latest_checkpoint} --data_dir {args.data_dir} --output_dir {args.output_dir} --additional_epochs {args.additional_epochs} --learning_rate {args.learning_rate} --batch_size {args.batch_size}"
                print(f"\nRunning: {cmd}")
                os.system(cmd)
                
            elif choice == "2":
                # Choose specific checkpoint
                print("\nAvailable checkpoints:")
                for i, checkpoint in enumerate(checkpoints, 1):
                    print(f"  {i}. {checkpoint}")
                
                try:
                    idx = int(input(f"\nEnter checkpoint number (1-{len(checkpoints)}): ")) - 1
                    if 0 <= idx < len(checkpoints):
                        selected_checkpoint = checkpoints[idx]
                        print(f"Resuming from: {selected_checkpoint}")
                        cmd = f"python resume_training.py --checkpoint_path {selected_checkpoint} --data_dir {args.data_dir} --output_dir {args.output_dir} --additional_epochs {args.additional_epochs} --learning_rate {args.learning_rate} --batch_size {args.batch_size}"
                        print(f"\nRunning: {cmd}")
                        os.system(cmd)
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
                    
            elif choice == "3":
                # Start fresh training
                print("Starting fresh training...")
                cmd = f"python train.py --data_dir {args.data_dir} --output_dir {args.output_dir} --epochs {args.additional_epochs} --learning_rate {args.learning_rate} --batch_size {args.batch_size}"
                print(f"\nRunning: {cmd}")
                os.system(cmd)
                
            else:
                print("Invalid choice.")

if __name__ == "__main__":
    main() 