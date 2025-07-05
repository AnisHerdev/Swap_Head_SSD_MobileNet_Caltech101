import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDHead, DefaultBoxGenerator
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.ssd import SSD, SSDScoringHead, SSDClassificationHead
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple, Union


class MobileNetFeatureExtractor(nn.Module):
    """
    Feature extractor using VGG backbone from SSD320
    """
    def __init__(self, pretrained=True):
        super(MobileNetFeatureExtractor, self).__init__()
        
        # Load SSD320 with VGG backbone
        self.ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT if pretrained else None)
        
        # Extract the backbone (feature extractor) from SSD
        self.backbone = self.ssd_model.backbone
        
        # Remove the detection head from SSD
        self._remove_detection_head()
    
    def _remove_detection_head(self):
        """Remove the detection head from SSD model"""
        if hasattr(self.ssd_model, 'head'):
            del self.ssd_model.head
        if hasattr(self.ssd_model, 'anchor_generator'):
            del self.ssd_model.anchor_generator
    
    def forward(self, x):
        """
        Forward pass through the feature extractor
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        Returns:
            features: Dictionary of feature maps at different scales
        """
        features = {}
        
        # Use the backbone's forward method to get features
        # The SSD backbone returns a dict of features at different scales
        backbone_features = self.backbone(x)
        
        # Extract the features we need
        # The backbone returns features with keys like '0', '1', '2', etc.
        # We'll map them to our layer names
        feature_keys = list(backbone_features.keys())
        
        if len(feature_keys) >= 4:
            # Map backbone features to our layer names
            features['layer1'] = backbone_features[feature_keys[0]]  # First layer
            features['layer2'] = backbone_features[feature_keys[1]]  # Second layer
            features['layer3'] = backbone_features[feature_keys[2]]  # Third layer
            features['layer4'] = backbone_features[feature_keys[3]]  # Fourth layer
        elif len(feature_keys) >= 1:
            # If we have fewer layers, use the deepest one for all
            deepest_features = backbone_features[feature_keys[-1]]
            features['layer1'] = deepest_features
            features['layer2'] = deepest_features
            features['layer3'] = deepest_features
            features['layer4'] = deepest_features
        else:
            # Fallback: create a simple feature map
            # This should not happen with a proper backbone
            raise ValueError("No features extracted from backbone")
        
        return features


class BridgingLayers(nn.Module):
    """
    Bridging layers to convert detection features to classification format
    """
    def __init__(self, input_channels=256, output_size=1280):
        super(BridgingLayers, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Adjust the feature size to match MobileNet classifier input
        self.feature_adapter = nn.Sequential(
            nn.Linear(input_channels, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    
    def forward(self, features):
        """
        Convert detection features to classification format
        Args:
            features: Dictionary of feature maps from detection backbone
        Returns:
            adapted_features: Features ready for classification head
        """
        # Use the deepest features (layer4) for classification
        x = features['layer4']
        
        # Global average pooling
        x = self.global_pool(x)
        x = self.flatten(x)
        
        # Adapt features to classification format
        x = self.feature_adapter(x)
        
        return x


class MobileNetClassifier(nn.Module):
    """
    MobileNet classifier head
    """
    def __init__(self, num_classes=1000, dropout=0.2):
        super(MobileNetClassifier, self).__init__()
        
        # Load MobileNet classifier
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        
        # Replace the classifier with our custom one
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through MobileNet classifier
        Args:
            x: Input features from bridging layers
        Returns:
            output: Classification logits
        """
        # The input should already be in the right format from bridging layers
        # We'll use the classifier part of MobileNet
        return self.mobilenet.classifier(x)


class HybridModel(nn.Module):
    """
    Hybrid model combining SSD320 feature extractor with MobileNet classifier
    """
    def __init__(self, num_classes=1000, pretrained=True):
        super(HybridModel, self).__init__()
        
        # Feature extractor (SSD320 backbone)
        self.feature_extractor = MobileNetFeatureExtractor(pretrained=pretrained)
        
        # Bridging layers to adapt features
        self.bridging_layers = BridgingLayers(input_channels=256, output_size=1280)
        
        # Classifier (MobileNet head)
        self.classifier = MobileNetClassifier(num_classes=num_classes)
        
        # Initialize weights for new layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for newly added layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the hybrid model
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        Returns:
            output: Classification logits
        """
        # Extract features using SSD backbone
        features = self.feature_extractor(x)
        
        # Bridge features to classification format
        adapted_features = self.bridging_layers(features)
        
        # Classify using MobileNet head
        output = self.classifier(adapted_features)
        
        return output
    
    def freeze_feature_extractor(self):
        """Freeze the feature extractor for transfer learning"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_feature_extractor(self):
        """Unfreeze the feature extractor for fine-tuning"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True


def create_hybrid_model(num_classes=1000, pretrained=True):
    """
    Factory function to create the hybrid model
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    Returns:
        model: The hybrid model
    """
    model = HybridModel(num_classes=num_classes, pretrained=pretrained)
    return model


if __name__ == "__main__":
    # Test the hybrid model
    model = create_hybrid_model(num_classes=10)
    model.eval()  # Set model to eval mode to avoid BatchNorm1d error on batch size 1
    
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 320, 320)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model created successfully!") 