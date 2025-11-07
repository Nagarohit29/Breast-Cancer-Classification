"""
Confusion Matrix Generator for Breast Cancer Classification Models
Provides standardized confusion matrix visualization across all models
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score
)
import os
import pandas as pd
import json

class ConfusionMatrixGenerator:
    """Standardized confusion matrix generator for all models"""
    
    def __init__(self, model_name, class_names=['Benign', 'Malignant']):
        self.model_name = model_name
        self.class_names = class_names
        
    def generate_confusion_matrix(self, y_true, y_pred, save_dir, normalize=False):
        """
        Generate and save confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            save_dir: Directory to save the confusion matrix
            normalize: Whether to normalize the confusion matrix
        
        Returns:
            cm: Confusion matrix array
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, 
                       annot=True, 
                       fmt='.2%',
                       cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       cbar_kws={'label': 'Percentage'})
            title = f'{self.model_name} - Normalized Confusion Matrix'
            filename = 'confusion_matrix_normalized.png'
        else:
            sns.heatmap(cm, 
                       annot=True, 
                       fmt='d',
                       cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       cbar_kws={'label': 'Count'})
            title = f'{self.model_name} - Confusion Matrix'
            filename = 'confusion_matrix.png'
            
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Save figure
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def generate_detailed_metrics(self, y_true, y_pred, save_dir):
        """
        Generate detailed classification metrics and save as text and JSON
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_dir: Directory to save the metrics
        
        Returns:
            metrics_dict: Dictionary containing all metrics
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate detailed metrics
        metrics_dict = {
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'accuracy': float(np.sum(np.diag(cm)) / np.sum(cm)),
            'precision': {
                'benign': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
                'malignant': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'macro_avg': float(precision_score(y_true, y_pred, average='macro')),
                'weighted_avg': float(precision_score(y_true, y_pred, average='weighted'))
            },
            'recall': {
                'benign': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'malignant': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'macro_avg': float(recall_score(y_true, y_pred, average='macro')),
                'weighted_avg': float(recall_score(y_true, y_pred, average='weighted'))
            },
            'f1_score': {
                'benign': float(2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) > 0 else 0.0,
                'malignant': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
                'macro_avg': float(f1_score(y_true, y_pred, average='macro')),
                'weighted_avg': float(f1_score(y_true, y_pred, average='weighted'))
            },
            'specificity': {
                'benign': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'malignant': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            },
            'sensitivity': {
                'benign': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'malignant': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            }
        }
        
        # Save metrics as JSON
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'detailed_metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Generate and save classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"Classification Report for {self.model_name}\n")
            f.write("=" * 50 + "\n")
            f.write(report)
        
        # Print summary
        print(f"\n📊 Detailed Metrics for {self.model_name}:")
        print(f"   Accuracy: {metrics_dict['accuracy']:.4f}")
        print(f"   Precision (Macro): {metrics_dict['precision']['macro_avg']:.4f}")
        print(f"   Recall (Macro): {metrics_dict['recall']['macro_avg']:.4f}")
        print(f"   F1-Score (Macro): {metrics_dict['f1_score']['macro_avg']:.4f}")
        
        # Calculate and display AUC if possible
        try:
            from sklearn.metrics import roc_auc_score
            # Note: This assumes binary classification - would need y_prob for AUC
            print(f"   📈 AUC calculation requires probability scores - check evaluation_metrics.py")
        except ImportError:
            pass
        
        return metrics_dict
    
    def generate_all_confusion_matrices(self, y_true, y_pred, save_dir):
        """
        Generate both normalized and regular confusion matrices
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_dir: Directory to save matrices
        
        Returns:
            tuple: (regular_cm, normalized_cm)
        """
        print(f"🎯 Generating confusion matrices for {self.model_name}...")
        
        # Generate regular confusion matrix
        cm_regular = self.generate_confusion_matrix(y_true, y_pred, save_dir, normalize=False)
        
        # Generate normalized confusion matrix
        cm_normalized = self.generate_confusion_matrix(y_true, y_pred, save_dir, normalize=True)
        
        # Generate detailed metrics
        metrics = self.generate_detailed_metrics(y_true, y_pred, save_dir)
        
        print(f"✅ Confusion matrices saved to: {save_dir}")
        
        return cm_regular, cm_normalized, metrics