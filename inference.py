#!/usr/bin/env python3
"""
Inference script for the hybrid model (SSD320 + MobileNet)
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from models.hybrid_model import create_hybrid_model
from utils.training_utils import load_trained_model


class HybridModelInference:
    """
    Class for making predictions with the trained hybrid model
    """
    def __init__(self, model_path: str, class_mapping_path: str, 
                 device: torch.device = None, input_size: int = 320):
        """
        Initialize the inference class
        Args:
            model_path: Path to the trained model checkpoint
            class_mapping_path: Path to the class mapping JSON file
            device: Device to run inference on
            input_size: Input image size
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)
        
        # Create model
        self.model = create_hybrid_model(num_classes=self.num_classes, pretrained=False)
        
        # Load trained weights
        self.model = load_trained_model(self.model, model_path, self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully!")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {list(self.class_mapping.keys())}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for inference
        Args:
            image_path: Path to the image file
        Returns:
            tensor: Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(self, image_path: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Make prediction on a single image
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
        Returns:
            predictions: List of predictions with class names and probabilities
        """
        # Preprocess image
        tensor = self.preprocess_image(image_path)
        tensor = tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        predictions = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            class_name = self.idx_to_class[class_idx]
            probability = top_probs[0][i].item()
            
            predictions.append({
                'class_name': class_name,
                'class_idx': class_idx,
                'probability': probability
            })
        
        return predictions
    
    def predict_batch(self, image_paths: List[str], top_k: int = 5) -> List[List[Dict[str, any]]]:
        """
        Make predictions on a batch of images
        Args:
            image_paths: List of image file paths
            top_k: Number of top predictions to return per image
        Returns:
            predictions: List of predictions for each image
        """
        predictions = []
        
        for image_path in image_paths:
            pred = self.predict(image_path, top_k)
            predictions.append(pred)
        
        return predictions
    
    def predict_with_confidence(self, image_path: str, confidence_threshold: float = 0.5) -> Optional[Dict[str, any]]:
        """
        Make prediction with confidence threshold
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence threshold
        Returns:
            prediction: Best prediction if above threshold, None otherwise
        """
        predictions = self.predict(image_path, top_k=1)
        
        if predictions[0]['probability'] >= confidence_threshold:
            return predictions[0]
        else:
            return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Inference with Hybrid Model (SSD320 + MobileNet)')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--class_mapping_path', type=str, required=True,
                       help='Path to the class mapping JSON file')
    
    # Input arguments
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to a single image for prediction')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory containing images for batch prediction')
    parser.add_argument('--input_size', type=int, default=320,
                       help='Input image size (default: 320)')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file to save predictions (JSON format)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to return (default: 5)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions (default: 0.5)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    return parser.parse_args()


def setup_device(args):
    """Setup device for inference"""
    if args.cpu:
        device = torch.device('cpu')
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    return device


def get_image_paths(image_dir: str) -> List[str]:
    """Get all image paths from a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    
    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(image_dir, filename))
    
    return sorted(image_paths)


def save_predictions(predictions: List[Dict[str, any]], output_file: str):
    """Save predictions to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to: {output_file}")


def print_predictions(predictions: List[Dict[str, any]], image_path: str = None):
    """Print predictions in a formatted way"""
    if image_path:
        print(f"\nPredictions for: {image_path}")
    print("-" * 50)
    
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['class_name']}: {pred['probability']:.4f} ({pred['probability']*100:.2f}%)")


def main():
    """Main inference function"""
    args = parse_args()
    
    # Setup device
    device = setup_device(args)
    
    # Initialize inference class
    inference = HybridModelInference(
        model_path=args.model_path,
        class_mapping_path=args.class_mapping_path,
        device=device,
        input_size=args.input_size
    )
    
    # Single image prediction
    if args.image_path:
        print(f"Making prediction on: {args.image_path}")
        
        if not os.path.exists(args.image_path):
            print(f"Error: Image file not found: {args.image_path}")
            return
        
        predictions = inference.predict(args.image_path, args.top_k)
        print_predictions(predictions, args.image_path)
        
        # Save predictions if output file specified
        if args.output_file:
            save_predictions(predictions, args.output_file)
    
    # Batch prediction
    elif args.image_dir:
        print(f"Making batch predictions on directory: {args.image_dir}")
        
        if not os.path.exists(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            return
        
        image_paths = get_image_paths(args.image_dir)
        print(f"Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            print("No image files found in the directory")
            return
        
        # Make predictions
        all_predictions = []
        for image_path in image_paths:
            predictions = inference.predict(image_path, args.top_k)
            all_predictions.append({
                'image_path': image_path,
                'predictions': predictions
            })
            
            # Print predictions
            print_predictions(predictions, image_path)
        
        # Save all predictions if output file specified
        if args.output_file:
            save_predictions(all_predictions, args.output_file)
    
    else:
        print("Error: Please specify either --image_path or --image_dir")
        return
    
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main() 