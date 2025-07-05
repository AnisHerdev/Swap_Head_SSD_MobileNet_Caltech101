# Caltech 101 Training Guide

This guide will help you train the hybrid model (SSD320 + MobileNet) on the Caltech 101 dataset.

## 📊 About Caltech 101

Caltech 101 is a classic computer vision dataset containing:
- **101 object categories** (plus 1 background category)
- **~9,000 images** total
- **~40-800 images per category**
- **High-quality, real-world images**
- **Perfect for object classification tasks**

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)
```bash
# Run the complete pipeline (download + prepare + train)
python run_caltech101_training.py
```

### Option 2: Step-by-Step
```bash
# Step 1: Download and prepare dataset
python download_caltech101.py

# Step 2: Train the model
python train_caltech101.py --data_dir ./caltech101_processed --output_dir ./caltech101_output
```

## 📁 Dataset Structure

After running the preparation script, your dataset will be organized as:
```
caltech101_processed/
├── train/
│   ├── accordion/
│   ├── airplane/
│   ├── anchor/
│   └── ... (all categories)
├── val/
│   ├── accordion/
│   ├── airplane/
│   └── ... (all categories)
└── test/
    ├── accordion/
    ├── airplane/
    └── ... (all categories)
```

## ⚙️ Training Configuration

### Default Settings
- **Input size**: 320x320 pixels
- **Batch size**: 16
- **Learning rate**: 0.001
- **Epochs**: 50 (with early stopping)
- **Optimizer**: AdamW
- **Loss**: CrossEntropyLoss
- **Data augmentation**: Random horizontal flip, rotation, color jitter

### Custom Training
```bash
python train_caltech101.py \
    --data_dir ./caltech101_processed \
    --output_dir ./caltech101_output \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --weight_decay 1e-4 \
    --early_stopping_patience 20
```

## 📈 Expected Performance

### Model Architecture
- **Feature Extractor**: SSD320 VGG backbone
- **Bridging Layers**: Global pooling + feature adaptation
- **Classifier**: MobileNet head
- **Total Parameters**: ~15-20M

### Expected Results
- **Training Time**: 2-4 hours (depending on GPU)
- **Validation Accuracy**: 85-95% (depending on training)
- **Test Accuracy**: 80-90%
- **Memory Usage**: ~4-8GB GPU memory

## 🔧 Advanced Options

### Transfer Learning
```bash
# Freeze feature extractor initially
python train_caltech101.py --freeze_backbone

# Then fine-tune the entire model
python train_caltech101.py --unfreeze_backbone
```

### Data Augmentation
The model uses the following augmentations:
- Random horizontal flip (p=0.5)
- Random rotation (±10 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization (ImageNet stats)

### Learning Rate Scheduling
- **ReduceLROnPlateau**: Reduces LR when validation accuracy plateaus
- **Factor**: 0.5 (halves the learning rate)
- **Patience**: 5 epochs

## 📊 Monitoring Training

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir ./caltech101_output/logs

# Open in browser: http://localhost:6006
```

### Training Metrics
- Training/validation loss
- Training/validation accuracy
- Learning rate progression
- Confusion matrix (after training)

## 🎯 Inference

### Single Image
```bash
python inference.py \
    --model_path ./caltech101_output/best_model.pth \
    --class_mapping_path ./caltech101_output/class_mapping.json \
    --image_path /path/to/image.jpg
```

### Batch Inference
```bash
python inference.py \
    --model_path ./caltech101_output/best_model.pth \
    --class_mapping_path ./caltech101_output/class_mapping.json \
    --image_dir /path/to/images/ \
    --output_file predictions.json
```

## 📁 Output Files

After training, you'll find:
```
caltech101_output/
├── best_model.pth              # Trained model checkpoint
├── class_mapping.json          # Class name to index mapping
├── training_metrics.json       # Training history
├── training_summary.json       # Final results summary
├── evaluation_results.json     # Test set results
├── confusion_matrix.png        # Confusion matrix visualization
└── logs/                       # TensorBoard logs
    ├── events.out.tfevents...
    └── ...
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train_caltech101.py --batch_size 8
   
   # Or use CPU
   python train_caltech101.py --cpu
   ```

2. **Slow Training**
   ```bash
   # Increase number of workers
   python train_caltech101.py --num_workers 8
   
   # Use smaller input size
   python train_caltech101.py --input_size 224
   ```

3. **Poor Performance**
   - Check dataset quality
   - Increase training epochs
   - Adjust learning rate
   - Try different data augmentation

### Performance Optimization

1. **Memory Optimization**
   ```python
   # Enable gradient checkpointing
   model.gradient_checkpointing_enable()
   ```

2. **Speed Optimization**
   ```python
   # Use mixed precision training
   scaler = torch.cuda.amp.GradScaler()
   ```

## 📚 Dataset Information

### Caltech 101 Categories
The dataset includes 101 object categories such as:
- Animals: accordion, airplane, anchor, ant, barrel, bass, beaver, binocular, bonsai, brain, brass, butterfly, camera, cannon, car_side, ceiling_fan, cellphone, chair, chandelier, cougar_face, cow, crayfish, crocodile_head, cup, dalmatian, dollar_bill, dolphin, dragonfly, electric_guitar, elephant, emu, euphonium, ewer, face, ferry, flamenco, flamingo_head, garfield, gerenuk, gramophone, grand_piano, hawksbill, headphone, hedgehog, helicopter, ibis, inline_skate, joshua_tree, kangaroo, ketch, lamp, laptop, leopards, llama, lobster, lotus, mandolin, mayfly, menorah, metronome, minaret, motorbike, nautilus, octopus, okapi, pagoda, panda, pigeon, pizza, platypus, pyramid, revolver, rhino, rooster, saxophone, schooner, scissors, scorpion, sea_horse, snoopy, soccer_ball, stapler, starfish, stegosaurus, stop_sign, strawberry, sunflower, tick, trilobite, umbrella, watch, water_lilly, wheelchair, wild_cat, windsor_chair, wrench, yin_yang

### Dataset Statistics
- **Total images**: ~9,000
- **Images per category**: 40-800
- **Image sizes**: Variable (typically 200-400 pixels)
- **Format**: JPEG, PNG
- **Quality**: High-quality, real-world images

## 🎉 Success Tips

1. **Start with default settings** - they're optimized for Caltech 101
2. **Monitor training** - use TensorBoard to track progress
3. **Use early stopping** - prevents overfitting
4. **Save checkpoints** - resume training if needed
5. **Evaluate thoroughly** - check confusion matrix for insights

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your dataset structure
3. Ensure all dependencies are installed
4. Check GPU memory availability

Happy training! 🚀 