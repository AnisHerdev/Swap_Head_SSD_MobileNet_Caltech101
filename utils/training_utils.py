import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from tqdm import tqdm
import time


class TrainingMetrics:
    """
    Class to track training metrics
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
    
    def update(self, train_loss: float, val_loss: float, 
               train_acc: float, val_acc: float, lr: float):
        """Update metrics"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
    
    def get_best_epoch(self) -> int:
        """Get the epoch with the best validation accuracy"""
        return np.argmax(self.val_accuracies)
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file"""
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_epoch': self.get_best_epoch()
        }
        
        # Convert any numpy types to native Python types
        metrics = convert_numpy_types(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy
    Args:
        outputs: Model outputs
        targets: Ground truth labels
    Returns:
        accuracy: Accuracy score
    """
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return correct / total


def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate various metrics
    Args:
        outputs: Model outputs
        targets: Ground truth labels
    Returns:
        metrics: Dictionary of metrics
    """
    _, predicted = torch.max(outputs.data, 1)
    
    # Convert to numpy for sklearn metrics
    predicted_np = predicted.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(targets_np, predicted_np)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, predicted_np, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def train_epoch(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer, 
                device: torch.device) -> Tuple[float, float]:
    """
    Train for one epoch
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc="Training")):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_outputs.append(outputs)
        all_targets.append(targets)
    
    # Calculate accuracy
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    accuracy = calculate_accuracy(all_outputs, all_targets)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy


def validate_epoch(model: nn.Module, val_loader: torch.utils.data.DataLoader,
                  criterion: nn.Module, device: torch.device) -> Tuple[float, float, Dict[str, float]]:
    """
    Validate for one epoch
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
        metrics: Additional metrics
    """
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(val_loader, desc="Validation")):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_outputs.append(outputs)
            all_targets.append(targets)
    
    # Calculate metrics
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    accuracy = calculate_accuracy(all_outputs, all_targets)
    metrics = calculate_metrics(all_outputs, all_targets)
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy, metrics


def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader, num_epochs: int,
                learning_rate: float = 0.001, weight_decay: float = 1e-4,
                device: torch.device = None, save_dir: str = './checkpoints',
                early_stopping_patience: int = 10) -> Dict[str, Any]:
    """
    Train the hybrid model
    Args:
        model: The hybrid model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device to train on
        save_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping
    Returns:
        training_info: Dictionary containing training information
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    
    # Metrics tracking
    metrics = TrainingMetrics()
    
    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics
        metrics.update(train_loss, val_loss, train_acc, val_acc, current_lr)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Additional Metrics: {val_metrics}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss
            }, os.path.join(save_dir, 'best_model.pth'))
            
            print(f"New best model saved! Validation Accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final metrics
    metrics.save_metrics(os.path.join(save_dir, 'training_metrics.json'))
    writer.close()
    
    # Load best model
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    return {
        'model': model,
        'best_val_acc': best_val_acc,
        'metrics': metrics,
        'checkpoint_path': best_model_path
    }


def evaluate_model(model: nn.Module, test_loader: torch.utils.data.DataLoader,
                  device: torch.device, class_names: List[str] = None,
                  save_dir: str = './results') -> Dict[str, Any]:
    """
    Evaluate the trained model
    Args:
        model: The trained model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of class names
        save_dir: Directory to save results
    Returns:
        results: Dictionary containing evaluation results
    """
    model.eval()
    all_outputs = []
    all_targets = []
    all_predictions = []
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing"):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_predictions.append(predicted.cpu())
    
    # Concatenate all results
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Calculate metrics
    accuracy = calculate_accuracy(all_outputs, all_targets)
    metrics = calculate_metrics(all_outputs, all_targets)
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets.numpy(), all_predictions.numpy())
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'accuracy': accuracy,
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions.numpy().tolist(),
        'targets': all_targets.numpy().tolist()
    }
    
    # Save results to JSON
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    if class_names:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Metrics: {metrics}")
    
    return results


def load_trained_model(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load a trained model from checkpoint
    Args:
        model: The model architecture
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
    Returns:
        model: The loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Training utilities loaded successfully!") 