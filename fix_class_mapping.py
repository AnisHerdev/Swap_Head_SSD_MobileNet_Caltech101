#!/usr/bin/env python3
"""
Fix class mapping by removing unwanted categories
"""

import json
import os

def fix_class_mapping():
    """Fix the class mapping by removing unwanted categories"""
    
    # Categories to exclude
    exclude_categories = {
        'BACKGROUND_Google',  # Background images
        'Faces',             # Face category (not object)
        'Faces_easy',        # Face category (not object)
        'Leopards',          # Duplicate of wild_cat
        'Motorbikes'         # Duplicate of motorbike
    }
    
    # Read current class mapping
    mapping_file = './caltech101_processed/class_mapping.json'
    with open(mapping_file, 'r') as f:
        current_mapping = json.load(f)
    
    print(f"Original classes: {len(current_mapping)}")
    print(f"Excluded categories: {exclude_categories}")
    
    # Remove excluded categories
    fixed_mapping = {}
    new_idx = 0
    for class_name, old_idx in current_mapping.items():
        if class_name not in exclude_categories:
            fixed_mapping[class_name] = new_idx
            new_idx += 1
    
    # Save fixed mapping
    with open(mapping_file, 'w') as f:
        json.dump(fixed_mapping, f, indent=2)
    
    print(f"Fixed classes: {len(fixed_mapping)}")
    print(f"Removed {len(current_mapping) - len(fixed_mapping)} unwanted categories")
    
    # Show first few classes
    print(f"First 10 classes: {list(fixed_mapping.keys())[:10]}")
    
    return fixed_mapping

if __name__ == "__main__":
    fixed_mapping = fix_class_mapping()
    print(f"\nClass mapping fixed! Now you can train with {len(fixed_mapping)} proper object categories.") 