# Hybrid Machine Learning Model: SSD320 + MobileNet

This project implements a hybrid machine learning model that combines the feature extractor (tail and body) of an SSD320 object detection model with the classifier head of a MobileNet image classification model. The hybrid approach leverages the robust feature extraction capabilities of SSD320 and the efficient classification capabilities of MobileNet.

## 🏗️ Architecture Overview

### Model Components

1. **Feature Extractor (SSD320 Backbone)**
   - Uses the convolutional layers from SSD320 for robust feature extraction
   - Extracts multi-scale features from different layers
   - Optimized for object detection tasks

2. **Bridging Layers**
   - Adapts detection features to classification format
   - Global average pooling to reduce spatial dimensions
   - Feature adaptation layers to match MobileNet classifier input

3. **Classifier Head (MobileNet)**
   - Uses MobileNet's efficient classifier
   - Lightweight and fast inference
   - Optimized for mobile and edge devices

### Architecture Flow

```
Input Image (320x320x3)
    ↓
SSD320 Feature Extractor
    ↓
Multi-scale Features
    ↓
Bridging Layers (Global Pooling + Adaptation)
    ↓
MobileNet Classifier
    ↓
Classification Output
```

## 📁 Project Structure

```
Research_work_Cursor/
├── models/
│   └── hybrid_model.py          # Main hybrid model implementation
├── utils/
│   ├── data_utils.py           # Data loading and preprocessing utilities
│   └── training_utils.py       # Training, evaluation, and metrics utilities
├── train.py                    # Main training script
├── inference.py                # Inference script for predictions
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Research_work_Cursor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; import torchvision; print('Installation successful!')"
   ```

## 📊 Dataset Preparation

### Directory Structure
Your dataset should be organized as follows:
```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image3.jpg
│   ├── image4.jpg
│   └── ...
└── ...
```

### Supported Formats
- Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- Automatic class detection from subdirectories
- Automatic train/validation/test split

## 🎯 Training

### Basic Training
```bash
python train.py --data_dir /path/to/your/dataset --output_dir ./output
```

### Advanced Training Options
```bash
python train.py \
    --data_dir /path/to/your/dataset \
    --output_dir ./output \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --input_size 320 \
    --early_stopping_patience 15
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | Required | Path to dataset directory |
| `--output_dir` | `./output` | Output directory for checkpoints |
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 0.001 | Learning rate |
| `--input_size` | 320 | Input image size |
| `--weight_decay` | 1e-4 | Weight decay for optimizer |
| `--early_stopping_patience` | 10 | Early stopping patience |

### Training Features

- **Automatic class detection** from dataset structure
- **Early stopping** to prevent overfitting
- **Learning rate scheduling** with ReduceLROnPlateau
- **TensorBoard logging** for training visualization
- **Model checkpointing** with best validation accuracy
- **Comprehensive metrics** tracking (accuracy, precision, recall, F1)

## 🔍 Inference

### Single Image Prediction
```bash
python inference.py \
    --model_path ./output/best_model.pth \
    --class_mapping_path ./output/class_mapping.json \
    --image_path /path/to/image.jpg
```

### Batch Prediction
```bash
python inference.py \
    --model_path ./output/best_model.pth \
    --class_mapping_path ./output/class_mapping.json \
    --image_dir /path/to/images/ \
    --output_file predictions.json
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | Required | Path to trained model checkpoint |
| `--class_mapping_path` | Required | Path to class mapping JSON |
| `--image_path` | None | Single image for prediction |
| `--image_dir` | None | Directory for batch prediction |
| `--top_k` | 5 | Number of top predictions to return |
| `--confidence_threshold` | 0.5 | Minimum confidence threshold |

## 📈 Model Performance

### Key Features
- **Efficient architecture** combining detection and classification strengths
- **Transfer learning** from pretrained models
- **Multi-scale feature extraction** from SSD320
- **Lightweight classification** with MobileNet
- **Robust bridging layers** for feature adaptation

### Expected Performance
- **Accuracy**: Comparable to standalone classification models
- **Speed**: Faster than full SSD320, slower than standalone MobileNet
- **Memory**: Moderate memory usage with efficient feature extraction
- **Flexibility**: Adaptable to various classification tasks

## 🔧 Customization

### Modifying the Model Architecture

1. **Change Feature Extractor:**
   ```python
   # In models/hybrid_model.py
   class MobileNetFeatureExtractor(nn.Module):
       def __init__(self, pretrained=True):
           # Modify to use different detection model backbone
           pass
   ```

2. **Modify Bridging Layers:**
   ```python
   # In models/hybrid_model.py
   class BridgingLayers(nn.Module):
       def __init__(self, input_channels=2048, output_size=1280):
           # Adjust feature adaptation layers
           pass
   ```

3. **Change Classifier:**
   ```python
   # In models/hybrid_model.py
   class MobileNetClassifier(nn.Module):
       def __init__(self, num_classes=1000, dropout=0.2):
           # Replace with different classification head
           pass
   ```

### Adding Custom Loss Functions

```python
# In utils/training_utils.py
def custom_loss_function(outputs, targets):
    # Implement your custom loss
    pass
```

## 📊 Monitoring and Visualization

### TensorBoard Integration
```bash
tensorboard --logdir ./output/logs
```

### Training Metrics
- Training/validation loss curves
- Accuracy progression
- Learning rate scheduling
- Confusion matrix visualization

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 8`
   - Use smaller input size: `--input_size 224`
   - Enable gradient checkpointing

2. **Slow Training**
   - Increase number of workers: `--num_workers 8`
   - Use mixed precision training
   - Optimize data loading pipeline

3. **Poor Performance**
   - Check dataset quality and balance
   - Adjust learning rate and weight decay
   - Try different model configurations

### Performance Optimization

1. **Memory Optimization:**
   ```python
   # Enable gradient checkpointing
   model.gradient_checkpointing_enable()
   ```

2. **Speed Optimization:**
   ```python
   # Use mixed precision training
   scaler = torch.cuda.amp.GradScaler()
   ```

## 📚 API Reference

### Main Classes

#### `HybridModel`
```python
model = create_hybrid_model(num_classes=10, pretrained=True)
```

#### `HybridModelInference`
```python
inference = HybridModelInference(
    model_path='./model.pth',
    class_mapping_path='./class_mapping.json'
)
predictions = inference.predict('image.jpg')
```

### Key Functions

#### Training
```python
from utils.training_utils import train_model, evaluate_model
training_info = train_model(model, train_loader, val_loader, num_epochs=50)
results = evaluate_model(model, test_loader, device)
```

#### Data Loading
```python
from utils.data_utils import create_data_loaders
train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size=32)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch and torchvision for the deep learning framework
- SSD320 and MobileNet architectures for inspiration
- The open-source community for various utilities and tools

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the maintainers
- Check the documentation and examples

---

**Happy Training! 🚀** 