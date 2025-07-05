#!/usr/bin/env python3
"""
Script to prepare existing Caltech 101 dataset for training
"""

import os
import shutil
import random
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image
import json


def prepare_dataset(data_dir="./caltech-101", output_dir="./caltech101_processed", 
                   train_split=0.7, val_split=0.2, test_split=0.1, min_images=10):
    """
    Prepare the dataset by organizing it into train/val/test splits
    Args:
        data_dir: Directory containing the original Caltech 101 dataset
        output_dir: Directory to save the processed dataset
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        min_images: Minimum number of images required per class
    """
    print("Preparing Caltech 101 dataset for training...")
    print(f"Source directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all category folders
    categories = []
    # Categories to exclude (background, non-object categories)
    exclude_categories = {
        'BACKGROUND_Google',  # Background images
        'Faces',             # Face category (not object)
        'Faces_easy',        # Face category (not object)
        'Leopards',          # Duplicate of wild_cat
        'Motorbikes'         # Duplicate of motorbike
    }
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Skip excluded categories
            if item in exclude_categories:
                print(f"Skipping excluded category: {item}")
                continue
                
            # Count images in this category
            image_files = [f for f in os.listdir(item_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if len(image_files) >= min_images:
                categories.append(item)
    
    print(f"Found {len(categories)} categories with at least {min_images} images")
    print(f"Categories: {categories[:10]}...")  # Show first 10 categories
    
    # Process each category
    total_images = 0
    for category in tqdm(categories, desc="Processing categories"):
        category_path = os.path.join(data_dir, category)
        
        # Get all image files
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate split indices
        n_images = len(image_files)
        n_train = int(n_images * train_split)
        n_val = int(n_images * val_split)
        
        # Split images
        train_images = image_files[:n_train]
        val_images = image_files[n_train:n_train + n_val]
        test_images = image_files[n_train + n_val:]
        
        # Create category directories
        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)
        test_category_dir = os.path.join(test_dir, category)
        
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(val_category_dir, exist_ok=True)
        os.makedirs(test_category_dir, exist_ok=True)
        
        # Copy images to respective directories
        for img_file in train_images:
            src = os.path.join(category_path, img_file)
            dst = os.path.join(train_category_dir, img_file)
            shutil.copy2(src, dst)
        
        for img_file in val_images:
            src = os.path.join(category_path, img_file)
            dst = os.path.join(val_category_dir, img_file)
            shutil.copy2(src, dst)
        
        for img_file in test_images:
            src = os.path.join(category_path, img_file)
            dst = os.path.join(test_category_dir, img_file)
            shutil.copy2(src, dst)
        
        total_images += n_images
    
    print(f"\nDataset preparation completed!")
    print(f"Total images processed: {total_images}")
    print(f"Train categories: {len(os.listdir(train_dir))}")
    print(f"Val categories: {len(os.listdir(val_dir))}")
    print(f"Test categories: {len(os.listdir(test_dir))}")


def verify_dataset(data_dir):
    """Verify the prepared dataset"""
    print("\nVerifying dataset...")
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    # Count train images
    for category in os.listdir(train_dir):
        category_path = os.path.join(train_dir, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            total_train += len(images)
    
    # Count val images
    for category in os.listdir(val_dir):
        category_path = os.path.join(val_dir, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            total_val += len(images)
    
    # Count test images
    for category in os.listdir(test_dir):
        category_path = os.path.join(test_dir, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            total_test += len(images)
    
    print(f"Dataset verification:")
    print(f"  Train images: {total_train}")
    print(f"  Val images: {total_val}")
    print(f"  Test images: {total_test}")
    print(f"  Total images: {total_train + total_val + total_test}")
    
    # Check a few sample images
    print("\nChecking sample images...")
    sample_categories = list(os.listdir(train_dir))[:3]
    
    for category in sample_categories:
        category_path = os.path.join(train_dir, category)
        images = [f for f in os.listdir(category_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if images:
            sample_image = os.path.join(category_path, images[0])
            try:
                img = Image.open(sample_image)
                print(f"  {category}: {img.size} ({img.mode})")
            except Exception as e:
                print(f"  {category}: Error reading image - {e}")


def save_class_mapping(data_dir, output_file='class_mapping.json'):
    """Save class mapping to JSON file"""
    classes = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))])
    class_mapping = {cls: idx for idx, cls in enumerate(classes)}
    
    with open(output_file, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"\nClass mapping saved to {output_file}")
    print(f"Found {len(classes)} classes")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Prepare Caltech 101 dataset for training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./caltech-101',
                       help='Directory containing the original Caltech 101 dataset')
    parser.add_argument('--output_dir', type=str, default='./caltech101_processed',
                       help='Directory to save the processed dataset')
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Fraction of data for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Fraction of data for validation')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Fraction of data for testing')
    parser.add_argument('--min_images', type=int, default=10,
                       help='Minimum number of images required per class')
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Source directory not found at {args.data_dir}")
        print("Please make sure the Caltech 101 dataset is in the correct location.")
        return
    
    # Prepare dataset
    prepare_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        min_images=args.min_images
    )
    
    # Verify dataset
    verify_dataset(args.output_dir)
    
    # Save class mapping (use train directory to get actual classes)
    save_class_mapping(os.path.join(args.output_dir, 'train'), os.path.join(args.output_dir, 'class_mapping.json'))
    
    print(f"\nDataset ready for training!")
    print(f"Use the following command to train:")
    print(f"python train_caltech101.py --data_dir {args.output_dir} --output_dir ./caltech101_output")


if __name__ == "__main__":
    main() 