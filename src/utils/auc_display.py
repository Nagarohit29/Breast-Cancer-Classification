"""
AUC Display Utility for Breast Cancer Classification Models
Provides prominent AUC reporting and interpretation
"""

def display_auc_results(metrics, model_name="Model"):
    """
    Display AUC results prominently with interpretation
    
    Args:
        metrics: Dictionary containing evaluation metrics
        model_name: Name of the model for display
    """
    print(f"\n{'='*60}")
    print(f"🎯 AUC RESULTS FOR {model_name.upper()}")
    print(f"{'='*60}")
    
    if 'advanced_metrics' in metrics and metrics['advanced_metrics']:
        auc_roc = metrics['advanced_metrics'].get('auc_roc')
        avg_precision = metrics['advanced_metrics'].get('average_precision')
        
        if auc_roc is not None:
            print(f"📈 AUC-ROC Score: {auc_roc:.4f}")
            
            # Interpretation
            if auc_roc >= 0.95:
                interpretation = "🌟 OUTSTANDING - Excellent discrimination"
                performance_level = "Outstanding"
            elif auc_roc >= 0.90:
                interpretation = "🔥 EXCELLENT - Very good discrimination"
                performance_level = "Excellent"
            elif auc_roc >= 0.80:
                interpretation = "✅ GOOD - Acceptable discrimination"
                performance_level = "Good"
            elif auc_roc >= 0.70:
                interpretation = "📊 FAIR - Some discrimination"
                performance_level = "Fair"
            elif auc_roc >= 0.60:
                interpretation = "⚠️  POOR - Limited discrimination"
                performance_level = "Poor"
            else:
                interpretation = "❌ VERY POOR - No discrimination"
                performance_level = "Very Poor"
            
            print(f"   {interpretation}")
            print(f"   Performance Level: {performance_level}")
            
            # Medical context interpretation
            print(f"\n🏥 Medical AI Context:")
            if auc_roc >= 0.90:
                print("   • Suitable for clinical decision support")
                print("   • High confidence in predictions")
            elif auc_roc >= 0.80:
                print("   • Suitable for screening assistance")
                print("   • Moderate confidence in predictions")
            elif auc_roc >= 0.70:
                print("   • May assist in preliminary screening")
                print("   • Requires careful validation")
            else:
                print("   • Not suitable for clinical use")
                print("   • Requires significant improvement")
        else:
            print("❌ AUC-ROC Score: Not available")
        
        if avg_precision is not None:
            print(f"\n📊 Average Precision Score: {avg_precision:.4f}")
            if avg_precision >= 0.90:
                print("   🌟 Excellent precision-recall performance")
            elif avg_precision >= 0.80:
                print("   ✅ Good precision-recall performance")
            elif avg_precision >= 0.70:
                print("   📊 Fair precision-recall performance")
            else:
                print("   ⚠️  Poor precision-recall performance")
        else:
            print("❌ Average Precision Score: Not available")
    
    else:
        print("❌ Advanced metrics not available")
        print("   Please check model evaluation implementation")
    
    # Display basic metrics for context
    if 'basic_metrics' in metrics:
        basic = metrics['basic_metrics']
        print(f"\n📋 Basic Metrics Summary:")
        print(f"   Accuracy:   {basic.get('accuracy', 'N/A'):.4f}" if isinstance(basic.get('accuracy'), (int, float)) else "   Accuracy:   N/A")
        print(f"   Precision:  {basic.get('precision', 'N/A'):.4f}" if isinstance(basic.get('precision'), (int, float)) else "   Precision:  N/A")
        print(f"   Recall:     {basic.get('recall', 'N/A'):.4f}" if isinstance(basic.get('recall'), (int, float)) else "   Recall:     N/A")
        print(f"   F1-Score:   {basic.get('f1_score', 'N/A'):.4f}" if isinstance(basic.get('f1_score'), (int, float)) else "   F1-Score:   N/A")
    
    print(f"{'='*60}")

def create_auc_summary_report(all_model_metrics, save_path="results/auc_summary_report.md"):
    """
    Create a comprehensive AUC summary report for all models
    
    Args:
        all_model_metrics: Dictionary of model_name -> metrics
        save_path: Path to save the summary report
    """
    import os
    from datetime import datetime
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    report_content = f"""# AUC Summary Report - Breast Cancer Classification

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 📊 AUC-ROC Performance Summary

| Model | AUC-ROC | Performance Level | Average Precision | Clinical Suitability |
|-------|---------|------------------|-------------------|---------------------|
"""
    
    for model_name, metrics in all_model_metrics.items():
        if metrics and 'advanced_metrics' in metrics and metrics['advanced_metrics']:
            auc_roc = metrics['advanced_metrics'].get('auc_roc', 'N/A')
            avg_precision = metrics['advanced_metrics'].get('average_precision', 'N/A')
            
            # Determine performance level
            if isinstance(auc_roc, (int, float)):
                if auc_roc >= 0.95:
                    perf_level = "Outstanding"
                    clinical = "Excellent for clinical use"
                elif auc_roc >= 0.90:
                    perf_level = "Excellent"
                    clinical = "Suitable for clinical support"
                elif auc_roc >= 0.80:
                    perf_level = "Good"
                    clinical = "Suitable for screening"
                elif auc_roc >= 0.70:
                    perf_level = "Fair"
                    clinical = "Limited clinical use"
                else:
                    perf_level = "Poor"
                    clinical = "Not suitable clinically"
                
                auc_display = f"{auc_roc:.4f}"
            else:
                perf_level = "N/A"
                clinical = "N/A"
                auc_display = "N/A"
            
            if isinstance(avg_precision, (int, float)):
                ap_display = f"{avg_precision:.4f}"
            else:
                ap_display = "N/A"
            
            report_content += f"| {model_name} | {auc_display} | {perf_level} | {ap_display} | {clinical} |\n"
        else:
            report_content += f"| {model_name} | N/A | N/A | N/A | No metrics available |\n"
    
    report_content += f"""
## 🎯 Performance Analysis

### Top Performing Models (by AUC-ROC)
"""
    
    # Sort models by AUC-ROC
    valid_aucs = []
    for model_name, metrics in all_model_metrics.items():
        if (metrics and 'advanced_metrics' in metrics and 
            metrics['advanced_metrics'] and 
            isinstance(metrics['advanced_metrics'].get('auc_roc'), (int, float))):
            valid_aucs.append((model_name, metrics['advanced_metrics']['auc_roc']))
    
    valid_aucs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model_name, auc_score) in enumerate(valid_aucs, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
        report_content += f"{i}. {emoji} **{model_name}**: AUC-ROC = {auc_score:.4f}\n"
    
    if not valid_aucs:
        report_content += "No valid AUC scores available yet.\n"
    
    report_content += f"""
## 🏥 Clinical Interpretation Guide

### AUC-ROC Score Interpretation:
- **0.95-1.00**: Outstanding - Excellent discrimination, suitable for clinical deployment
- **0.90-0.94**: Excellent - Very good discrimination, suitable for clinical decision support
- **0.80-0.89**: Good - Acceptable discrimination, suitable for screening assistance
- **0.70-0.79**: Fair - Some discrimination ability, may assist in preliminary screening
- **0.60-0.69**: Poor - Limited discrimination, requires significant improvement
- **0.50-0.59**: Very Poor - No better than random guessing

### Medical AI Context:
- **Cancer Detection**: High sensitivity (recall) is crucial - missing cancer cases is costly
- **False Positives**: While concerning, false positives are generally less critical than false negatives
- **AUC Threshold**: For medical applications, typically require AUC ≥ 0.80 for clinical consideration

## 📈 Recommendations

Based on AUC performance:
"""
    
    if valid_aucs:
        best_model, best_auc = valid_aucs[0]
        report_content += f"- **Primary Recommendation**: {best_model} (AUC: {best_auc:.4f})\n"
        
        if len(valid_aucs) > 1:
            second_model, second_auc = valid_aucs[1]
            report_content += f"- **Alternative**: {second_model} (AUC: {second_auc:.4f})\n"
    
    report_content += """
- Consider ensemble methods for improved performance
- Validate on external datasets before clinical deployment
- Implement proper cross-validation for robust evaluation

---
*This report is automatically generated after model training completion.*
"""
    
    # Save the report
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"🎯 AUC Summary Report saved to: {save_path}")
    return save_path