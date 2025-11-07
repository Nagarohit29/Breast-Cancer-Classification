#!/usr/bin/env python3
"""
Generate enhanced confusion matrix and ROC curve visualizations for Model 1 (AlexNet and VGG16 with SVM)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_model_metrics(model_path):
    """Load metrics from JSON file"""
    with open(model_path, 'r') as f:
        return json.load(f)

def create_confusion_matrix_plot(cm, model_name, accuracy, save_path):
    """Create enhanced confusion matrix visualization matching Model 2 style"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'],
                ax=ax, cbar_kws={'label': 'Count'})
    
    # Add percentage annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j+0.5, i+0.7, f'({cm_percent[i,j]:.1f}%)', 
                          ha="center", va="center", color="red", fontweight='bold')
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add title and labels
    ax.set_title(f'{model_name} Confusion Matrix\nAccuracy: {accuracy:.4f}', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    
    # Add comprehensive metrics text box
    metrics_text = f'Precision: {precision:.4f}\nRecall: {recall:.4f}\nSpecificity: {specificity:.4f}\nF1-Score: {f1:.4f}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_curve_plot(fpr, tpr, auc_score, model_name, save_path):
    """Create enhanced ROC curve visualization matching Model 2 style"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=3, 
            label=f'ROC Curve (AUC = {auc_score:.4f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.5000)')
    
    # Set axis limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'{model_name} ROC Curve', fontsize=16, fontweight='bold')
    
    # Add legend and grid
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add AUC score annotation
    ax.text(0.6, 0.2, f'AUC = {auc_score:.4f}', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plot(alexnet_metrics, vgg16_metrics, save_path):
    """Create comparison plot for both models"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model names and colors
    models = ['AlexNet + SVM', 'VGG16 + SVM']
    colors = ['#1f77b4', '#ff7f0e']
    
    # 1. Accuracy comparison
    accuracies = [alexnet_metrics['accuracy'], vgg16_metrics['accuracy']]
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.7)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim([0.8, 1.0])
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. AUC comparison
    aucs = [alexnet_metrics['auc'], vgg16_metrics['auc']]
    bars2 = ax2.bar(models, aucs, color=colors, alpha=0.7)
    ax2.set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('AUC Score', fontsize=12)
    ax2.set_ylim([0.9, 1.0])
    
    # Add value labels on bars
    for bar, auc_val in zip(bars2, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{auc_val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. ROC curves comparison
    for i, (metrics, model, color) in enumerate(zip([alexnet_metrics, vgg16_metrics], models, colors)):
        fpr = metrics['roc_curve']['fpr']
        tpr = metrics['roc_curve']['tpr']
        auc_score = metrics['auc']
        ax3.plot(fpr, tpr, color=color, lw=2, label=f'{model} (AUC = {auc_score:.4f})')
    
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='Random Classifier')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate', fontsize=12)
    ax3.set_ylabel('True Positive Rate', fontsize=12)
    ax3.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax3.legend(loc="lower right", fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion matrices comparison
    cm1 = np.array(alexnet_metrics['confusion_matrix'])
    cm2 = np.array(vgg16_metrics['confusion_matrix'])
    
    # Normalize confusion matrices
    cm1_norm = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
    cm2_norm = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
    
    # Plot first confusion matrix
    im1 = ax4.imshow(cm1_norm, interpolation='nearest', cmap='Blues')
    ax4.set_title('Confusion Matrices Comparison', fontsize=14, fontweight='bold')
    
    # Add text annotations for first matrix
    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            ax4.text(j, i, f'{cm1[i,j]}\n({cm1_norm[i,j]:.2f})', 
                    ha="center", va="center", color="red" if i == j else "black", fontweight='bold')
    
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Benign', 'Malignant'])
    ax4.set_yticklabels(['Benign', 'Malignant'])
    ax4.set_xlabel('Predicted Label')
    ax4.set_ylabel('True Label')
    
    # Add model labels
    ax4.text(0.5, -0.3, 'AlexNet + SVM', transform=ax4.transAxes, ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate all visualizations"""
    # Define paths
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    results_dir = os.path.join(base_dir, "results", "model 1")
    models_dir = os.path.join(base_dir, "models")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model metrics
    alexnet_metrics = load_model_metrics(os.path.join(models_dir, "paper1_alexnet_svm", "metrics.json"))
    vgg16_metrics = load_model_metrics(os.path.join(models_dir, "paper1_vgg16_svm", "metrics.json"))
    
    print("Generating visualizations for Model 1...")
    
    # 1. Model 1a (AlexNet) Confusion Matrix
    alexnet_cm = np.array(alexnet_metrics['confusion_matrix'])
    create_confusion_matrix_plot(
        alexnet_cm, 
        "Model 1a (AlexNet + SVM)", 
        alexnet_metrics['accuracy'],
        os.path.join(results_dir, "model_1a_final_confusion_matrix.png")
    )
    print("[OK] Model 1a confusion matrix saved")
    
    # 2. Model 1b (VGG16) Confusion Matrix
    vgg16_cm = np.array(vgg16_metrics['confusion_matrix'])
    create_confusion_matrix_plot(
        vgg16_cm, 
        "Model 1b (VGG16 + SVM)", 
        vgg16_metrics['accuracy'],
        os.path.join(results_dir, "model_1b_final_confusion_matrix.png")
    )
    print("[OK] Model 1b confusion matrix saved")
    
    # 3. Model 1a (AlexNet) ROC Curve
    create_roc_curve_plot(
        alexnet_metrics['roc_curve']['fpr'],
        alexnet_metrics['roc_curve']['tpr'],
        alexnet_metrics['auc'],
        "Model 1a (AlexNet + SVM)",
        os.path.join(results_dir, "model_1a_final_roc_curve.png")
    )
    print("[OK] Model 1a ROC curve saved")
    
    # 4. Model 1b (VGG16) ROC Curve
    create_roc_curve_plot(
        vgg16_metrics['roc_curve']['fpr'],
        vgg16_metrics['roc_curve']['tpr'],
        vgg16_metrics['auc'],
        "Model 1b (VGG16 + SVM)",
        os.path.join(results_dir, "model_1b_final_roc_curve.png")
    )
    print("[OK] Model 1b ROC curve saved")
    
    # 5. Comparison Plot
    create_comparison_plot(
        alexnet_metrics, 
        vgg16_metrics,
        os.path.join(results_dir, "model1_comparison.png")
    )
    print("[OK] Model comparison plot saved")
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL 1 VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Model 1a (AlexNet + SVM):")
    print(f"  - Accuracy: {alexnet_metrics['accuracy']:.4f}")
    print(f"  - AUC: {alexnet_metrics['auc']:.4f}")
    print(f"  - Precision: {alexnet_metrics['precision']:.4f}")
    print(f"  - Recall: {alexnet_metrics['recall']:.4f}")
    print(f"  - F1-Score: {alexnet_metrics['f1_score']:.4f}")
    
    print(f"\nModel 1b (VGG16 + SVM):")
    print(f"  - Accuracy: {vgg16_metrics['accuracy']:.4f}")
    print(f"  - AUC: {vgg16_metrics['auc']:.4f}")
    print(f"  - Precision: {vgg16_metrics['precision']:.4f}")
    print(f"  - Recall: {vgg16_metrics['recall']:.4f}")
    print(f"  - F1-Score: {vgg16_metrics['f1_score']:.4f}")
    
    print(f"\nAll visualizations saved to: {results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
