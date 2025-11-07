import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
# Ensure repository root is on sys.path so `src` imports work when running scripts directly
project_root = Path(__file__).resolve().parents[1]
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.utils.path_resolver import resolve_breakhis_path
import joblib

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt


class SimpleCSVImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        self.class_to_idx = {'benign': 0, 'malignant': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        # Resolve to actual file under data_dir when possible
        path = resolve_breakhis_path(filename, self.data_dir)
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx.get(row.get('class', 'benign'), 0)
        return img, label


def get_feature_extractor(name='alexnet', device='cpu'):
    if name == 'alexnet':
        model = models.alexnet(pretrained=True)
        # return flattened conv outputs by removing classifier
        model.classifier = nn.Identity()
    elif name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = nn.Identity()
    else:
        raise ValueError('unknown backbone')
    model.eval()
    model.to(device)
    return model


def extract_features(dataloader, model, device='cpu'):
    features = []
    labels = []
    with torch.no_grad():
        for imgs, labs in tqdm(dataloader, desc='Extracting features', unit='batch'):
            imgs = imgs.to(device)
            out = model(imgs)
            # ensure flattened
            out = out.view(out.size(0), -1).cpu().numpy()
            features.append(out)
            labels.append(labs.numpy())
    features = np.vstack(features)
    labels = np.concatenate(labels)
    return features, labels


def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / 'data' / 'processed'
    base_image_dir = project_root / 'data' / 'raw'
    # Allow overrides from environment
    data_dir = os.getenv('DATA_DIR', None)
    magnification = os.getenv('MAGNIFICATION', 'all_mags')
    if data_dir:
        base_image_dir = Path(data_dir)

    # Strict-paper config (env-overridable)
    PCA_COMPONENTS = int(os.getenv('PCA_COMPONENTS', 512))
    SVM_KERNEL = os.getenv('SVM_KERNEL', 'rbf')
    SVM_C = float(os.getenv('SVM_C', 1.0))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_csv = str(processed_dir / f'train_{magnification}.csv')
    val_csv = str(processed_dir / f'validation_{magnification}.csv')

    # If CSVs missing, demo with random tensors
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print('WARNING: CSVs not found, using synthetic data for a quick demo')
        X_train = np.random.randn(100, 512)
        y_train = np.random.randint(0,2,100)
        X_val = np.random.randn(20, 512)
        y_val = np.random.randint(0,2,20)
        # Fix PCA components for demo
        demo_pca_components = min(PCA_COMPONENTS, 50)  # Use smaller number for demo
        pca = PCA(n_components=demo_pca_components)
        X_train_p = pca.fit_transform(X_train)
        X_val_p = pca.transform(X_val)
        clf = SVC(kernel=SVM_KERNEL, C=SVM_C, probability=True, class_weight='balanced')
        clf.fit(X_train_p, y_train)
        preds = clf.predict(X_val_p)
        pred_probs = clf.predict_proba(X_val_p)[:, 1]
        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, pred_probs)
        print(f'Demo accuracy: {acc:.4f}')
        print(f'Demo AUC: {auc:.4f}')
        return

    train_ds = SimpleCSVImageDataset(train_csv, str(base_image_dir), transform=transform)
    val_ds = SimpleCSVImageDataset(val_csv, str(base_image_dir), transform=transform)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    for backbone in ('alexnet', 'vgg16'):
        print(f'\n{"="*50}')
        print(f'Processing {backbone.upper()} backbone')
        print(f'{"="*50}')
        
        print(f'Extracting features with {backbone} on {device}...')
        model = get_feature_extractor(backbone, device)
        feats_train, y_train = extract_features(train_loader, model, device)
        feats_val, y_val = extract_features(val_loader, model, device)

        print(f'Training features shape: {feats_train.shape}')
        print(f'Validation features shape: {feats_val.shape}')
        print(f'Training labels distribution: {np.bincount(y_train)}')
        print(f'Validation labels distribution: {np.bincount(y_val)}')

        # PCA for dimensionality reduction
        pca = PCA(n_components=PCA_COMPONENTS)
        feats_train_p = pca.fit_transform(feats_train)
        feats_val_p = pca.transform(feats_val)

        scaler = StandardScaler()
        feats_train_p = scaler.fit_transform(feats_train_p)
        feats_val_p = scaler.transform(feats_val_p)

        clf = SVC(kernel=SVM_KERNEL, C=SVM_C, probability=True, class_weight='balanced')
        print('Training SVM...')
        clf.fit(feats_train_p, y_train)

        # Get predictions and probabilities
        preds = clf.predict(feats_val_p)
        pred_probs = clf.predict_proba(feats_val_p)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, pred_probs)
        
        # Confusion Matrix
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f'\n{backbone.upper()} Results:')
        print(f'Accuracy: {acc:.4f}')
        print(f'AUC: {auc:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall (Sensitivity): {recall:.4f}')
        print(f'Specificity: {specificity:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print(f'\nConfusion Matrix:')
        print(f'TN: {tn}, FP: {fp}')
        print(f'FN: {fn}, TP: {tp}')
        
        print(f'\nDetailed Classification Report:')
        print(classification_report(y_val, preds, target_names=['Benign', 'Malignant']))

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_val, pred_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {backbone.upper()}')
        plt.legend(loc="lower right")
        
        # Save ROC curve
        out_dir = project_root / 'models' / f'paper1_{backbone}_svm'
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save artifacts
        joblib.dump({'pca': pca, 'scaler': scaler, 'svm': clf}, str(out_dir / 'feature_svm_pipeline.joblib'))
        
        # Save metrics
        metrics = {
            'accuracy': float(acc),
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        }
        
        import json
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f'Saved pipeline and metrics to {out_dir}')


if __name__ == '__main__':
    main()
