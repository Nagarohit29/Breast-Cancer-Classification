"""
Common Evaluation Metrics for All PyTorch Models
Provides standardized evaluation functions and metrics calculation
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.results = {}
        
    def evaluate_model(self, model, test_loader, save_dir):
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            save_dir: Directory to save results
        
        Returns:
            dict: Complete evaluation results
        """
        print(f"\n📊 Evaluating {self.model_name}...")
        
        # Set model to evaluation mode
        model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Get model predictions
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
                
                if batch_idx % 10 == 0:
                    print(f"   Evaluated {batch_idx * len(data)}/{len(test_loader.dataset)} samples")
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        
        # Save detailed results
        self._save_detailed_results(metrics, save_dir)
        
        # Generate and save plots
        self._generate_evaluation_plots(y_true, y_pred, y_prob, save_dir)
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'basic_metrics': {},
            'advanced_metrics': {},
            'classification_report': {},
            'sample_info': {
                'total_samples': len(y_true),
                'positive_samples': int(np.sum(y_true)),
                'negative_samples': int(len(y_true) - np.sum(y_true))
            }
        }
        
        # Basic metrics
        metrics['basic_metrics'] = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='binary')),
            'recall': float(recall_score(y_true, y_pred, average='binary')),
            'f1_score': float(f1_score(y_true, y_pred, average='binary')),
            'specificity': float(self._calculate_specificity(y_true, y_pred))
        }
        
        # Advanced metrics
        if len(np.unique(y_true)) == 2:  # Binary classification
            try:
                auc_roc = roc_auc_score(y_true, y_prob)
                avg_precision = average_precision_score(y_true, y_prob)
                
                metrics['advanced_metrics'] = {
                    'auc_roc': float(auc_roc),
                    'average_precision': float(avg_precision)
                }
            except ValueError as e:
                print(f"⚠️  Warning: Could not calculate AUC metrics: {e}")
                metrics['advanced_metrics'] = {
                    'auc_roc': None,
                    'average_precision': None
                }
        
        # Classification report
        try:
            class_report = classification_report(y_true, y_pred, output_dict=True)
            metrics['classification_report'] = class_report
        except Exception as e:
            print(f"⚠️  Warning: Could not generate classification report: {e}")
            metrics['classification_report'] = {}
        
        return metrics
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (True Negative Rate)"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if (tn + fp) == 0:
            return 0.0
        return tn / (tn + fp)
    
    def _save_detailed_results(self, metrics, save_dir):
        """Save detailed evaluation results to JSON"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save complete metrics
        results_file = os.path.join(save_dir, f'{self.model_name}_evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save summary metrics to CSV
        summary_data = {
            'Model': [self.model_name],
            'Accuracy': [metrics['basic_metrics']['accuracy']],
            'Precision': [metrics['basic_metrics']['precision']],
            'Recall': [metrics['basic_metrics']['recall']],
            'F1-Score': [metrics['basic_metrics']['f1_score']],
            'Specificity': [metrics['basic_metrics']['specificity']]
        }
        
        if metrics['advanced_metrics']['auc_roc'] is not None:
            summary_data['AUC-ROC'] = [metrics['advanced_metrics']['auc_roc']]
            summary_data['Avg Precision'] = [metrics['advanced_metrics']['average_precision']]
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(save_dir, f'{self.model_name}_metrics_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"✅ Results saved to {save_dir}")
        print(f"   - Detailed: {results_file}")
        print(f"   - Summary: {summary_file}")
    
    def _generate_evaluation_plots(self, y_true, y_pred, y_prob, save_dir):
        """Generate and save evaluation plots"""
        
        # ROC Curve (separate file)
        if len(np.unique(y_true)) == 2:
            self._plot_roc_curve(y_true, y_prob, save_dir)
            
            # Precision-Recall Curve (separate file)
            self._plot_precision_recall_curve(y_true, y_prob, save_dir)
        
        # Class distribution plot (separate file)
        self._plot_class_distribution(y_true, y_pred, save_dir)
    
    def _plot_roc_curve(self, y_true, y_prob, save_dir):
        """Plot and save ROC curve"""
        plt.figure(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{self.model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        roc_file = os.path.join(save_dir, f'{self.model_name}_roc_curve.png')
        plt.savefig(roc_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   - ROC Curve: {roc_file}")
    
    def _plot_precision_recall_curve(self, y_true, y_prob, save_dir):
        """Plot and save Precision-Recall curve"""
        plt.figure(figsize=(8, 6))
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'{self.model_name} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        pr_file = os.path.join(save_dir, f'{self.model_name}_precision_recall.png')
        plt.savefig(pr_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   - Precision-Recall: {pr_file}")
    
    def _plot_class_distribution(self, y_true, y_pred, save_dir):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True labels distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        ax1.bar(['Benign', 'Malignant'], counts_true, color=['lightblue', 'lightcoral'], alpha=0.7)
        ax1.set_title('True Labels Distribution', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=11)
        
        # Predicted labels distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        ax2.bar(['Benign', 'Malignant'], counts_pred, color=['lightgreen', 'lightsalmon'], alpha=0.7)
        ax2.set_title('Predicted Labels Distribution', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=11)
        
        plt.suptitle(f'Class Distribution - {self.model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        dist_file = os.path.join(save_dir, f'{self.model_name}_class_distribution.png')
        plt.savefig(dist_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   - Class Distribution: {dist_file}")

def print_evaluation_summary(metrics):
    """Print a formatted summary of evaluation metrics"""
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION SUMMARY: {metrics['model_name']}")
    print(f"{'='*60}")
    
    print(f"\n📈 Basic Metrics:")
    basic = metrics['basic_metrics']
    print(f"   Accuracy:    {basic['accuracy']:.4f}")
    print(f"   Precision:   {basic['precision']:.4f}")
    print(f"   Recall:      {basic['recall']:.4f}")
    print(f"   F1-Score:    {basic['f1_score']:.4f}")
    print(f"   Specificity: {basic['specificity']:.4f}")
    
    if metrics['advanced_metrics']['auc_roc'] is not None:
        print(f"\n🎯 Advanced Metrics:")
        advanced = metrics['advanced_metrics']
        print(f"   AUC-ROC:     {advanced['auc_roc']:.4f}")
        print(f"   Avg Precision: {advanced['average_precision']:.4f}")
    
    sample_info = metrics['sample_info']
    print(f"\n📊 Sample Information:")
    print(f"   Total Samples:    {sample_info['total_samples']}")
    print(f"   Positive (Malignant): {sample_info['positive_samples']}")
    print(f"   Negative (Benign):    {sample_info['negative_samples']}")
    
    print(f"\n{'='*60}")