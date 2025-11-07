"""
Common Confusion Matrix Generator for All PyTorch Models
Provides standardized confusion matrix generation and visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import json
from datetime import datetime

class ConfusionMatrixGenerator:
    """Comprehensive confusion matrix generator and visualizer"""
    
    def __init__(self, model_name, class_names=None):
        self.model_name = model_name
        self.class_names = class_names or ['Benign', 'Malignant']
        
    def generate_confusion_matrix(self, model, test_loader, device, save_dir):
        """
        Generate and save confusion matrix with comprehensive analysis
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            device: PyTorch device
            save_dir: Directory to save results
            
        Returns:
            dict: Confusion matrix analysis results
        """
        print(f"\n🔍 Generating Confusion Matrix for {self.model_name}...")
        
        # Get predictions
        y_true, y_pred, y_prob = self._get_predictions(model, test_loader, device)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate detailed metrics from confusion matrix
        cm_analysis = self._analyze_confusion_matrix(cm, y_true, y_pred)
        
        # Create visualizations
        self._create_confusion_matrix_plots(cm, save_dir)
        
        # Save detailed analysis
        self._save_confusion_matrix_analysis(cm_analysis, cm, save_dir)
        
        # Print summary
        self._print_confusion_matrix_summary(cm, cm_analysis)
        
        return cm_analysis
    
    def _get_predictions(self, model, test_loader, device):
        """Get model predictions on test data"""
        model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                
                outputs = model(data)
                
                # Handle different output formats
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    # Multi-class output
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    probs = probabilities[:, 1].cpu().numpy() if outputs.shape[1] == 2 else probabilities.cpu().numpy()
                else:
                    # Binary output
                    probabilities = torch.sigmoid(outputs.squeeze())
                    predictions = (probabilities > 0.5).long()
                    probs = probabilities.cpu().numpy()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probs if hasattr(probs, '__len__') else [probs])
                all_labels.extend(target.cpu().numpy())
        
        return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)
    
    def _analyze_confusion_matrix(self, cm, y_true, y_pred):
        """Analyze confusion matrix and calculate detailed metrics"""
        
        # Extract values from confusion matrix
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate metrics
        total = len(y_true)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # False positive rate and false negative rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        analysis = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'derived_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'f1_score': float(f1_score),
                'false_positive_rate': float(fpr),
                'false_negative_rate': float(fnr)
            },
            'sample_counts': {
                'total_samples': int(total),
                'actual_positive': int(np.sum(y_true)),
                'actual_negative': int(total - np.sum(y_true)),
                'predicted_positive': int(np.sum(y_pred)),
                'predicted_negative': int(total - np.sum(y_pred))
            }
        }
        
        return analysis
    
    def _create_confusion_matrix_plots(self, cm, save_dir):
        """Create and save confusion matrix visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot 1: Standard confusion matrix with counts
        self._plot_confusion_matrix_counts(cm, save_dir)
        
        # Plot 2: Normalized confusion matrix (percentages)
        self._plot_confusion_matrix_normalized(cm, save_dir)
        
        # Plot 3: Detailed confusion matrix with both counts and percentages
        self._plot_confusion_matrix_detailed(cm, save_dir)
    
    def _plot_confusion_matrix_counts(self, cm, save_dir):
        """Plot confusion matrix with raw counts"""
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {self.model_name}\n(Raw Counts)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add text annotations for clarity
        plt.text(0.5, -0.1, 'TN: True Negative, FP: False Positive\nFN: False Negative, TP: True Positive', 
                ha='center', transform=plt.gca().transAxes, fontsize=10, style='italic')
        
        counts_file = os.path.join(save_dir, f'{self.model_name}_confusion_matrix_counts.png')
        plt.savefig(counts_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   - Counts Matrix: {counts_file}")
    
    def _plot_confusion_matrix_normalized(self, cm, save_dir):
        """Plot normalized confusion matrix with percentages"""
        plt.figure(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Percentage'})
        
        plt.title(f'Confusion Matrix - {self.model_name}\n(Normalized Percentages)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        normalized_file = os.path.join(save_dir, f'{self.model_name}_confusion_matrix_normalized.png')
        plt.savefig(normalized_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   - Normalized Matrix: {normalized_file}")
    
    def _plot_confusion_matrix_detailed(self, cm, save_dir):
        """Plot detailed confusion matrix with both counts and percentages"""
        plt.figure(figsize=(10, 8))
        
        # Create annotations with both counts and percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percentage = cm_normalized[i, j]
                row.append(f'{count}\n({percentage:.1%})')
            annotations.append(row)
        
        sns.heatmap(cm, annot=np.array(annotations), fmt='', cmap='RdYlBu_r',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Detailed Confusion Matrix - {self.model_name}\n(Counts and Percentages)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add performance metrics as text
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        total = cm.sum()
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.3f}  |  Precision: {precision:.3f}  |  Recall: {recall:.3f}'
        plt.text(0.5, -0.05, metrics_text, ha='center', transform=plt.gca().transAxes, 
                fontsize=10, fontweight='bold')
        
        detailed_file = os.path.join(save_dir, f'{self.model_name}_confusion_matrix_detailed.png')
        plt.savefig(detailed_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   - Detailed Matrix: {detailed_file}")
    
    def _save_confusion_matrix_analysis(self, analysis, cm, save_dir):
        """Save confusion matrix analysis to JSON file"""
        
        # Add raw confusion matrix to analysis
        analysis['raw_confusion_matrix'] = cm.tolist()
        
        # Save to JSON
        analysis_file = os.path.join(save_dir, f'{self.model_name}_confusion_matrix_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"   - Analysis JSON: {analysis_file}")
    
    def _print_confusion_matrix_summary(self, cm, analysis):
        """Print formatted confusion matrix summary"""
        print(f"\n{'='*50}")

    def generate_confusion_matrix_from_arrays(self, y_true, y_pred, save_dir):
        """
        Generate confusion matrix and analysis from numpy arrays of true/pred labels.

        Args:
            y_true: array-like of true labels
            y_pred: array-like of predicted labels
            save_dir: directory to save outputs

        Returns:
            dict: confusion matrix analysis
        """
        print(f"\n🔍 Generating Confusion Matrix from arrays for {self.model_name}...")

        # Ensure numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Build confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Analyze and save
        cm_analysis = self._analyze_confusion_matrix(cm, y_true, y_pred)
        self._create_confusion_matrix_plots(cm, save_dir)
        self._save_confusion_matrix_analysis(cm_analysis, cm, save_dir)
        self._print_confusion_matrix_summary(cm, cm_analysis)

        return cm_analysis
        print(f"🔍 CONFUSION MATRIX SUMMARY: {self.model_name}")
        print(f"{'='*50}")
        
        # Raw confusion matrix
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        print(f"\n📊 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"              Benign  Malignant")
        print(f"   Actual Benign    {tn:4d}      {fp:4d}")
        print(f"        Malignant   {fn:4d}      {tp:4d}")
        
        # Derived metrics
        metrics = analysis['derived_metrics']
        print(f"\n📈 Performance Metrics:")
        print(f"   Accuracy:     {metrics['accuracy']:.4f}")
        print(f"   Precision:    {metrics['precision']:.4f}")
        print(f"   Recall:       {metrics['recall']:.4f}")
        print(f"   Specificity:  {metrics['specificity']:.4f}")
        print(f"   F1-Score:     {metrics['f1_score']:.4f}")
        
        # Error rates
        print(f"\n⚠️  Error Rates:")
        print(f"   False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"   False Negative Rate: {metrics['false_negative_rate']:.4f}")
        
        print(f"\n{'='*50}")

def create_combined_confusion_matrices(model_results, save_dir):
    """Create a combined visualization of all model confusion matrices"""
    
    if not model_results:
        print("⚠️  No model results provided for combined confusion matrix")
        return
    
    num_models = len(model_results)
    fig, axes = plt.subplots(2, (num_models + 1) // 2, figsize=(5 * ((num_models + 1) // 2), 10))
    
    if num_models == 1:
        axes = [axes]
    elif num_models <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    class_names = ['Benign', 'Malignant']
    
    for idx, (model_name, result) in enumerate(model_results.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Extract confusion matrix from results
        if 'confusion_matrix' in result:
            cm_data = result['confusion_matrix']
            cm = np.array([[cm_data['true_negative'], cm_data['false_positive']],
                          [cm_data['false_negative'], cm_data['true_positive']]])
        else:
            # Create dummy data if not available
            cm = np.array([[0, 0], [0, 0]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar=False)
        
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    # Hide unused subplots
    for idx in range(num_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrices - All Models Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    combined_file = os.path.join(save_dir, 'all_models_confusion_matrices_comparison.png')
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Combined confusion matrices saved: {combined_file}")