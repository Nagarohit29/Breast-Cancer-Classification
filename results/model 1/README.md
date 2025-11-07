# Model 1 Results - AlexNet+SVM and VGG16+SVM

This directory contains the evaluation results for **Model 1**, which consists of two variants:
- **Model 1a**: AlexNet + SVM
- **Model 1b**: VGG16 + SVM

## Generated Visualizations (Model 2 Style - DPI 300)

### Individual Model Results

#### Model 1a (AlexNet + SVM)
- **Confusion Matrix**: `model_1a_final_confusion_matrix.png` (10x8, DPI 300)
- **ROC Curve**: `model_1a_final_roc_curve.png` (10x8, DPI 300)
- **Metrics**:
  - Accuracy: 0.8937 (89.37%)
  - AUC: 0.9687
  - Precision: 0.9726 (97.26%)
  - Recall: 0.8727 (87.27%)
  - F1-Score: 0.9200

#### Model 1b (VGG16 + SVM)
- **Confusion Matrix**: `model_1b_final_confusion_matrix.png` (10x8, DPI 300)
- **ROC Curve**: `model_1b_final_roc_curve.png` (10x8, DPI 300)
- **Metrics**:
  - Accuracy: 0.9087 (90.87%)
  - AUC: 0.9679
  - Precision: 0.9686 (96.86%)
  - Recall: 0.8988 (89.88%)
  - F1-Score: 0.9324

### Comparison Visualization
- **Combined Comparison**: `model1_comparison.png`
  - Side-by-side comparison of both models
  - Includes accuracy comparison, AUC comparison, ROC curves overlay, and confusion matrices

## Performance Summary

| Model | Accuracy | AUC | Precision | Recall | F1-Score |
|-------|----------|-----|-----------|--------|----------|
| Model 1a (AlexNet+SVM) | 89.37% | 0.9687 | 97.26% | 87.27% | 0.9200 |
| Model 1b (VGG16+SVM) | **90.87%** | 0.9679 | 96.86% | **89.88%** | **0.9324** |

## Key Findings

1. **Better Overall Performance**: Model 1b (VGG16+SVM) achieves higher accuracy (90.87% vs 89.37%)
2. **Similar AUC Scores**: Both models have excellent AUC scores (≈0.97), indicating strong discriminative ability
3. **Higher Recall**: VGG16+SVM has better recall (89.88% vs 87.27%), meaning it catches more malignant cases
4. **Slightly Lower Precision**: AlexNet+SVM has marginally higher precision (97.26% vs 96.86%)
5. **Better F1-Score**: VGG16+SVM achieves a better balance with F1-Score of 0.9324

## File Descriptions

**Main Visualizations (Model 2 Style):**
- `model_1a_final_confusion_matrix.png` - Confusion matrix for Model 1a (AlexNet+SVM) with enhanced styling
- `model_1a_final_roc_curve.png` - ROC curve for Model 1a (AlexNet+SVM) with enhanced styling
- `model_1b_final_confusion_matrix.png` - Confusion matrix for Model 1b (VGG16+SVM) with enhanced styling
- `model_1b_final_roc_curve.png` - ROC curve for Model 1b (VGG16+SVM) with enhanced styling
- `model1_comparison.png` - Combined comparison of both models

**Legacy Files:**
- `alexnet_confusion_matrix.png` - Previous AlexNet visualization
- `alexnet_roc_curve.png` - Previous AlexNet ROC curve
- `vgg16_confusion_matrix.png` - Previous VGG16 visualization
- `vgg16_roc_curve.png` - Previous VGG16 ROC curve

**Metrics Data:**
- `paper1_alexnet_svm_evaluation_results.json` - Detailed metrics for AlexNet+SVM
- `paper1_vgg16_svm_evaluation_results.json` - Detailed metrics for VGG16+SVM
- `predictions.csv` - Per-sample predictions
- `metrics.json` - Consolidated metrics

## Recommendation

**Model 1b (VGG16+SVM)** is recommended for deployment due to:
- Higher overall accuracy
- Better recall (important for medical diagnosis to minimize false negatives)
- Better balanced performance (higher F1-score)
