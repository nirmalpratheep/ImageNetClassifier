"""
Comprehensive visualization and metrics module for ImageNet-1K training.

This module provides:
- Training/validation curves plotting
- Confusion matrix visualization
- Classification metrics calculation
- Model performance analysis
- Learning rate visualization
"""
import os
import os

# If Jupyter inline backend leaks into CLI, fix it
if os.environ.get("MPLBACKEND", "").startswith("module://matplotlib_inline"):
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_fscore_support,
    accuracy_score, top_k_accuracy_score
)
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ImageNet-1K class names (1000 classes)
# For visualization purposes, we'll use generic class names since ImageNet-1K has 1000 classes
IMAGENET_CLASSES = [f"class_{i:03d}" for i in range(1000)]

# CIFAR-10 class names (for backward compatibility)
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CIFAR-100 class names (for backward compatibility)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


class MetricsCalculator:
    """Calculate comprehensive classification metrics."""
    
    def __init__(self, num_classes: int = 1000, class_names: List[str] = None):
        self.num_classes = num_classes
        if class_names is not None:
            self.class_names = class_names
        elif num_classes == 1000:
            self.class_names = IMAGENET_CLASSES
        elif num_classes == 100:
            self.class_names = CIFAR100_CLASSES
        else:
            self.class_names = CIFAR10_CLASSES
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive classification metrics."""
        
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Top-k accuracy
        if y_pred_proba is not None:
            try:
                # Create full label set to handle cases where validation set doesn't contain all classes
                all_labels = np.arange(self.num_classes)
                
                # Only calculate top-k if k <= number of classes
                if self.num_classes >= 5:
                    metrics['top_5_accuracy'] = top_k_accuracy_score(y_true, y_pred_proba, k=5, labels=all_labels)
                else:
                    metrics['top_5_accuracy'] = metrics['accuracy']  # Fallback for datasets with < 5 classes
                    
                if self.num_classes >= 3:
                    metrics['top_3_accuracy'] = top_k_accuracy_score(y_true, y_pred_proba, k=3, labels=all_labels)
                else:
                    metrics['top_3_accuracy'] = metrics['accuracy']  # Fallback for datasets with < 3 classes
                    
            except ValueError as e:
                print(f"Warning: Could not calculate top-k accuracy: {e}")
                print(f"Using regular accuracy as fallback")
                metrics['top_5_accuracy'] = metrics['accuracy']
                metrics['top_3_accuracy'] = metrics['accuracy']
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics.update({
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support
        })
        
        return metrics


class TrainingVisualizer:
    """Visualize training progress and model performance."""
    
    def __init__(self, save_dir: str = "./plots", class_names: List[str] = None, num_classes: int = 1000):
        self.save_dir = save_dir
        if class_names is not None:
            self.class_names = class_names
        elif num_classes == 1000:
            self.class_names = IMAGENET_CLASSES
        elif num_classes == 100:
            self.class_names = CIFAR100_CLASSES
        else:
            self.class_names = CIFAR10_CLASSES
        self.num_classes = num_classes
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_curves(self, train_losses: List[float], train_acc: List[float],
                           test_losses: List[float], test_acc: List[float],
                           learning_rates: List[float] = None, save_name: str = "training_curves"):
        """Plot training and validation curves."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, test_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, test_acc, 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate curve
        if learning_rates:
            axes[1, 0].plot(epochs, learning_rates, 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Learning Rate Schedule')
        
        # Generalization gap
        gap = np.array(train_acc) - np.array(test_acc)
        axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
        axes[1, 1].set_title('Generalization Gap (Train - Val)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Gap (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
        
        plt.show()
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_name: str = "confusion_matrix"):
        """Plot confusion matrix with class names."""
        
        cm = confusion_matrix(y_true, y_pred)
        
        # For ImageNet-1K (1000 classes), create a smaller, more manageable plot
        if self.num_classes == 1000:
            plt.figure(figsize=(20, 16))
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap without annotations for readability
            sns.heatmap(cm_normalized, annot=False, fmt='.3f', cmap='Blues',
                       xticklabels=False, yticklabels=False,
                       cbar_kws={'label': 'Normalized Count'})
            
            plt.title('Confusion Matrix (Normalized) - ImageNet-1K (1000 classes)', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label (1000 classes)', fontsize=12)
            plt.ylabel('True Label (1000 classes)', fontsize=12)
            
        else:
            # For smaller datasets (CIFAR-10/100), use the original detailed plot
            plt.figure(figsize=(12, 10))
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       cbar_kws={'label': 'Normalized Count'})
            
            plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            # Add raw counts as text
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})', 
                            ha='center', va='center', fontsize=8, color='red')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
        
    def plot_class_metrics(self, metrics: Dict, save_name: str = "class_metrics"):
        """Plot per-class precision, recall, and F1 scores."""
        
        precision = metrics['precision_per_class']
        recall = metrics['recall_per_class']
        f1 = metrics['f1_per_class']
        support = metrics['support_per_class']
        
        # For ImageNet-1K (1000 classes), create a summary plot instead of per-class
        if self.num_classes == 1000:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('ImageNet-1K Performance Metrics Summary', fontsize=16, fontweight='bold')
            
            # Histogram of precision scores
            axes[0, 0].hist(precision, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_title('Precision Distribution')
            axes[0, 0].set_xlabel('Precision')
            axes[0, 0].set_ylabel('Number of Classes')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axvline(np.mean(precision), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(precision):.3f}')
            axes[0, 0].legend()
            
            # Histogram of recall scores
            axes[0, 1].hist(recall, bins=50, alpha=0.7, color='orange', edgecolor='black')
            axes[0, 1].set_title('Recall Distribution')
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Number of Classes')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axvline(np.mean(recall), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(recall):.3f}')
            axes[0, 1].legend()
            
            # Histogram of F1 scores
            axes[1, 0].hist(f1, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[1, 0].set_title('F1 Score Distribution')
            axes[1, 0].set_xlabel('F1 Score')
            axes[1, 0].set_ylabel('Number of Classes')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axvline(np.mean(f1), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(f1):.3f}')
            axes[1, 0].legend()
            
            # Histogram of support (number of samples)
            axes[1, 1].hist(support, bins=50, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].set_title('Support Distribution')
            axes[1, 1].set_xlabel('Number of Samples')
            axes[1, 1].set_ylabel('Number of Classes')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axvline(np.mean(support), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(support):.1f}')
            axes[1, 1].legend()
            
        else:
            # For smaller datasets (CIFAR-10/100), use the original detailed plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
            
            x = np.arange(len(self.class_names))
            width = 0.25
            
            # Precision
            axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8)
            axes[0, 0].set_title('Precision per Class')
            axes[0, 0].set_ylabel('Precision')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(self.class_names, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Recall
            axes[0, 1].bar(x, recall, width, label='Recall', alpha=0.8, color='orange')
            axes[0, 1].set_title('Recall per Class')
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(self.class_names, rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # F1 Score
            axes[1, 0].bar(x + width, f1, width, label='F1 Score', alpha=0.8, color='green')
            axes[1, 0].set_title('F1 Score per Class')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(self.class_names, rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Support (number of samples)
            axes[1, 1].bar(x, support, alpha=0.8, color='purple')
            axes[1, 1].set_title('Support per Class')
            axes[1, 1].set_ylabel('Number of Samples')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(self.class_names, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class metrics saved to: {save_path}")
        
        plt.show()
        
    def plot_learning_rate_schedule(self, learning_rates: List[float], 
                                  save_name: str = "learning_rate_schedule"):
        """Plot learning rate schedule."""
        
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(learning_rates) + 1)
        plt.plot(epochs, learning_rates, 'b-', linewidth=2, marker='o', markersize=4)
        
        plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Add annotations for key points
        if len(learning_rates) > 1:
            plt.annotate(f'Start: {learning_rates[0]:.6f}', 
                        xy=(1, learning_rates[0]), xytext=(10, 10),
                        textcoords='offset points', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            plt.annotate(f'End: {learning_rates[-1]:.6f}', 
                        xy=(len(learning_rates), learning_rates[-1]), xytext=(10, -20),
                        textcoords='offset points', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate schedule saved to: {save_path}")
        
        plt.show()
        
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    metrics: Dict, save_name: str = "classification_report"):
        """Generate and save detailed classification report."""
        
        report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                    digits=4, output_dict=True)
        
        # Create a comprehensive report
        dataset_name = "ImageNet-1K" if self.num_classes == 1000 else "CIFAR-10" if self.num_classes == 10 else "CIFAR-100"
        report_text = f"""
{dataset_name} Classification Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

Overall Performance:
- Accuracy: {metrics['accuracy']:.4f}
- Top-3 Accuracy: {metrics.get('top_3_accuracy', 'N/A'):.4f}
- Top-5 Accuracy: {metrics.get('top_5_accuracy', 'N/A'):.4f}

Macro Averages:
- Precision: {metrics['precision_macro']:.4f}
- Recall: {metrics['recall_macro']:.4f}
- F1-Score: {metrics['f1_macro']:.4f}

Weighted Averages:
- Precision: {metrics['precision_weighted']:.4f}
- Recall: {metrics['recall_weighted']:.4f}
- F1-Score: {metrics['f1_weighted']:.4f}
"""

        # For ImageNet-1K, show only top/bottom performers instead of all classes
        if self.num_classes == 1000:
            report_text += f"""
Per-Class Performance (Top 20 Best and Worst Classes):
{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}
{'-'*60}
"""
            
            # Get top and bottom performers by F1 score
            f1_scores = metrics['f1_per_class']
            sorted_indices = np.argsort(f1_scores)[::-1]  # descending order
            
            # Top 20 performers
            for i in range(min(20, len(sorted_indices))):
                idx = sorted_indices[i]
                class_name = self.class_names[idx]
                report_text += f"{class_name:<12} {metrics['precision_per_class'][idx]:<10.4f} {metrics['recall_per_class'][idx]:<10.4f} {metrics['f1_per_class'][idx]:<10.4f} {int(metrics['support_per_class'][idx]):<8}\n"
            
            report_text += f"\n... (showing top 20 of {self.num_classes} classes) ...\n\n"
            
            # Bottom 20 performers
            report_text += f"Worst 20 Classes:\n"
            for i in range(max(0, len(sorted_indices) - 20), len(sorted_indices)):
                idx = sorted_indices[i]
                class_name = self.class_names[idx]
                report_text += f"{class_name:<12} {metrics['precision_per_class'][idx]:<10.4f} {metrics['recall_per_class'][idx]:<10.4f} {metrics['f1_per_class'][idx]:<10.4f} {int(metrics['support_per_class'][idx]):<8}\n"
        else:
            # For smaller datasets, show all classes
            report_text += f"""
Per-Class Performance:
{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}
{'-'*60}
"""
            
            for i, class_name in enumerate(self.class_names):
                report_text += f"{class_name:<12} {metrics['precision_per_class'][i]:<10.4f} {metrics['recall_per_class'][i]:<10.4f} {metrics['f1_per_class'][i]:<10.4f} {int(metrics['support_per_class'][i]):<8}\n"
        
        # Save report
        save_path = os.path.join(self.save_dir, f"{save_name}.txt")
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"Classification report saved to: {save_path}")
        print(report_text)
        
        return report_text


def evaluate_model_comprehensive(model, device, test_loader, criterion, class_names=None, num_classes=1000):
    """Comprehensive model evaluation with all predictions and probabilities."""
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    test_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            # Get predictions and probabilities
            probabilities = F.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    test_loss /= len(test_loader)
    
    return {
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probabilities),
        'test_loss': test_loss,
        'num_classes': num_classes
    }


def create_training_summary(train_losses, train_acc, test_losses, test_acc, 
                          learning_rates=None, save_dir="./plots", num_classes=1000):
    """Create a comprehensive training summary with all visualizations."""
    
    visualizer = TrainingVisualizer(save_dir, num_classes=num_classes)
    
    # Plot training curves
    visualizer.plot_training_curves(train_losses, train_acc, test_losses, test_acc, learning_rates)
    
    # Plot learning rate schedule if available
    if learning_rates:
        visualizer.plot_learning_rate_schedule(learning_rates)
    
    print("[OK] Training summary plots generated successfully!")
    print(f"[Output] All plots saved in: {save_dir}")


def create_evaluation_summary(model, device, test_loader, criterion, save_dir="./plots", num_classes=1000):
    """Create comprehensive evaluation summary with confusion matrix and metrics."""
    
    visualizer = TrainingVisualizer(save_dir, num_classes=num_classes)
    metrics_calc = MetricsCalculator(num_classes=num_classes)
    
    # Get comprehensive evaluation results
    eval_results = evaluate_model_comprehensive(model, device, test_loader, criterion, num_classes=num_classes)
    
    # Calculate metrics
    metrics = metrics_calc.calculate_metrics(
        eval_results['targets'], 
        eval_results['predictions'],
        eval_results['probabilities']
    )
    
    # Generate visualizations
    visualizer.plot_confusion_matrix(eval_results['targets'], eval_results['predictions'])
    visualizer.plot_class_metrics(metrics)
    visualizer.generate_classification_report(eval_results['targets'], eval_results['predictions'], metrics)
    
    print("[OK] Evaluation summary generated successfully!")
    print(f"[Output] All evaluation plots saved in: {save_dir}")
    
    return metrics, eval_results

