"""
Model 3: Ensemble CNN for Breast Cancer Classification
Combines ResNet-50 and VGG-16 with a fusion layer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Import custom utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.gpu_utils import setup_device, clear_gpu_memory, set_reproducible_training
from src.utils.path_resolver import resolve_breakhis_path
from src.utils.evaluation_metrics import ModelEvaluator, print_evaluation_summary
from src.utils.confusion_matrix import ConfusionMatrixGenerator
from src.utils.auc_display import display_auc_results
import random

class EnsembleCNN(nn.Module):
    """Ensemble model combining ResNet-50 and VGG-16"""
    
    def __init__(self, num_classes=2):
        super(EnsembleCNN, self).__init__()
        
        # ResNet-50 branch
        self.resnet_branch = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.resnet_features = nn.Sequential(*list(self.resnet_branch.children())[:-1])
        
        # VGG-16 branch
        self.vgg_branch = models.vgg16(pretrained=True)
        # Remove the classifier and keep only features
        self.vgg_features = self.vgg_branch.features
        
        # Adaptive pooling for VGG features
        self.vgg_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimensions
        resnet_feature_dim = 2048  # ResNet-50 output
        vgg_feature_dim = 512      # VGG-16 output after pooling
        
        # Freeze pre-trained features initially
        self._freeze_pretrained_features()
        
        # Feature fusion layers
        self.resnet_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(resnet_feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        self.vgg_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(vgg_feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        # Attention mechanism for feature fusion
        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.features_frozen = True
    
    def _freeze_pretrained_features(self):
        """Freeze pre-trained feature extractors"""
        for param in self.resnet_features.parameters():
            param.requires_grad = False
        for param in self.vgg_features.parameters():
            param.requires_grad = False
    
    def unfreeze_features(self):
        """Unfreeze pre-trained features for fine-tuning"""
        for param in self.resnet_features.parameters():
            param.requires_grad = True
        for param in self.vgg_features.parameters():
            param.requires_grad = True
        self.features_frozen = False
        print("[INFO] Pre-trained features unfrozen for fine-tuning")
    
    def forward(self, x):
        # ResNet-50 branch
        resnet_features = self.resnet_features(x)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        resnet_out = self.resnet_fc(resnet_features)
        
        # VGG-16 branch
        vgg_features = self.vgg_features(x)
        vgg_features = self.vgg_adaptive_pool(vgg_features)
        vgg_features = vgg_features.view(vgg_features.size(0), -1)
        vgg_out = self.vgg_fc(vgg_features)
        
        # Concatenate features
        combined_features = torch.cat([resnet_out, vgg_out], dim=1)
        
        # Attention-based fusion
        attention_weights = self.attention(combined_features)
        
        # Apply attention weights
        resnet_weighted = resnet_out * attention_weights[:, 0:1]
        vgg_weighted = vgg_out * attention_weights[:, 1:2]
        
        # Final fusion
        fused_features = torch.cat([resnet_weighted, vgg_weighted], dim=1)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output

class BreaKHisDataset(Dataset):
    """Custom dataset for BreaKHis breast cancer images"""
    
    def __init__(self, csv_file, transform=None, data_dir="datasets/breakhis"):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.data_dir = data_dir
        
        # Map class names to indices
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Map CSV path to actual file system path
        csv_path = row['filename']
        actual_path = resolve_breakhis_path(csv_path, self.data_dir)
        
        # Load image
        try:
            image = Image.open(actual_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {actual_path}: {e}")
            # Create a dummy image
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.class_to_idx[row['class']]
        
        return image, label
    
    def _map_csv_path_to_actual(self, csv_path):
        """Map CSV path format to actual file system path"""
        # Delegate to path_resolver for robust mapping
        return resolve_breakhis_path(csv_path, self.data_dir)

def _normalize_magnification(magnification):
    if magnification is None:
        return 'all_mags'
    if isinstance(magnification, int):
        return f"{magnification}X"
    mag = str(magnification).strip()
    if mag.lower() in ('all', 'all_mags', 'allmags'):
        return 'all_mags'
    if mag.endswith('X') or mag.endswith('x'):
        return mag.upper()
    return f"{mag}X"


def create_data_loaders(batch_size=16, data_dir=None, magnification="100X"):
    """Create data loaders with ensemble-optimized augmentation"""
    
    # Comprehensive train transforms for ensemble training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(25),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Resolve paths
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / 'data' / 'processed'
    base_image_dir = project_root / 'data' / 'raw'
    if data_dir:
        base_image_dir = Path(data_dir)

    magnification = _normalize_magnification(magnification)

    # Create datasets
    try:
        train_csv = str(processed_dir / f'train_{magnification}.csv')
        val_csv = str(processed_dir / f'validation_{magnification}.csv')
        test_csv = str(processed_dir / f'test_{magnification}.csv')

        train_dataset = BreaKHisDataset(csv_file=train_csv, transform=train_transform, data_dir=str(base_image_dir))
        val_dataset = BreaKHisDataset(csv_file=val_csv, transform=val_test_transform, data_dir=str(base_image_dir))
        test_dataset = BreaKHisDataset(csv_file=test_csv, transform=val_test_transform, data_dir=str(base_image_dir))

    except FileNotFoundError as e:
        print(f"[WARNING] CSV files not found: {e}")
        print("Creating minimal datasets for demonstration...")
        
        # Create minimal datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 224, 224),
            torch.randint(0, 2, (100,))
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.randn(20, 3, 224, 224),
            torch.randint(0, 2, (20,))
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.randn(30, 3, 224, 224),
            torch.randint(0, 2, (30,))
        )
    
    # Create data loaders (smaller batch size for ensemble model)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, num_epochs=30, fine_tune_after=20):
    """Train a single backbone with two-phase training: freeze then fine-tune.

    Returns trained model and history dict.
    """
    # Use label smoothing for robustness
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Initially only train classifier parameters (assumes backbone has .fc or .classifier)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs), eta_min=1e-6)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0
    best_state = None

    fine_tuning_started = False

    print(f"\n[INFO] Training backbone for {num_epochs} epochs (fine_tune_after={fine_tune_after})")

    for epoch in range(num_epochs):
        # switch to fine-tuning by unfreezing all params
        if epoch == fine_tune_after and any(not p.requires_grad for p in model.parameters()):
            print("[INFO] Unfreezing backbone for fine-tuning")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - fine_tune_after), eta_min=1e-7)
            fine_tuning_started = True

        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        from tqdm import tqdm
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{num_epochs}', unit='batch')):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += torch.sum(preds == target).item()
            total += data.size(0)

            if batch_idx % 50 == 0:
                phase = 'fine-tune' if fine_tuning_started else 'feature-extract'
                tqdm.write(f'  Epoch {epoch+1}/{num_epochs} [{phase}] batch {batch_idx} loss {loss.item():.4f}')

        train_loss = running_loss / max(1, total)
        train_acc = running_corrects / max(1, total)

        # validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item() * data.size(0)
                preds = outputs.argmax(dim=1)
                val_corrects += torch.sum(preds == target).item()
                val_total += data.size(0)

        val_loss = val_loss / max(1, val_total)
        val_acc = val_corrects / max(1, val_total)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['lr'].append(float(lr))

        print(f'Epoch {epoch+1}/{num_epochs} done. Train acc {train_acc:.4f} Val acc {val_acc:.4f} LR {lr:.6e}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            clear_gpu_memory()

    if best_state:
        model.load_state_dict(best_state)

    return model, history

def save_model_and_results(model, history, model_name, save_dir):
    """Save model weights and training history in organized folders."""
    model_dir = os.path.join("models", model_name)
    results_dir = os.path.join("results", model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}_model.pth")
    torch.save(model.state_dict(), model_path)

    history_path = os.path.join(save_dir, f"{model_name}_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # duplicate into results dir
    results_history_path = os.path.join(results_dir, f"{model_name}_training_history.json")
    with open(results_history_path, 'w') as f:
        json.dump(history, f, indent=2)

        print(f"[OK] Saved {model_name} to {model_path} and history to {history_path}")

def main():
    """Main training function for Model 3"""
    
    print("="*80)
    print("MODEL 3: Ensemble CNN (ResNet-50 + VGG-16) for Breast Cancer Classification")
    print("="*80)
    
    # Config (env overrides for strict paper matching)
    SEED = int(os.getenv('SEED', 42))
    TRAIN_EPOCHS = int(os.getenv('TRAIN_EPOCHS', 25))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 16))
    LR = float(os.getenv('LR', 1e-3))
    OPTIM = os.getenv('OPTIM', 'adamw')
    FINE_TUNE_AFTER = int(os.getenv('FINE_TUNE_AFTER', 20))

    # Deterministic seeding
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    try:
        set_reproducible_training(SEED)
    except Exception:
        pass

    # Setup device
    device, device_info = setup_device()

    # Model parameters
    model_name = "model3_ensemble_cnn"
    batch_size = BATCH_SIZE  # Smaller batch size for ensemble model
    num_epochs = TRAIN_EPOCHS
    fine_tune_after = FINE_TUNE_AFTER
    
    # Create save directories
    save_dir = os.path.join("results", model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Create data loaders
        print("\nLoading data...")
        train_loader, val_loader, test_loader = create_data_loaders(batch_size=batch_size)
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Train two independent backbones and ensemble by averaging probabilities
        backbones = ['resnet50', 'vgg16']
        trained_models = {}
        histories = {}

        for backbone in backbones:
            print(f"\n[INFO] Building and training backbone: {backbone}")
            if backbone == 'resnet50':
                # Build ResNet-50 and attach classifier head
                try:
                    from torchvision.models import ResNet50_Weights
                    weights = ResNet50_Weights.IMAGENET1K_V1
                    net = models.resnet50(weights=weights)
                except Exception:
                    net = models.resnet50(pretrained=True)

                num_features = net.fc.in_features
                net.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.2),
                    nn.Linear(256, 2)
                )
            else:
                # VGG-16
                try:
                    net = models.vgg16(pretrained=True)
                except Exception:
                    net = models.vgg16(pretrained=True)
                # Replace classifier with smaller head
                net.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(25088, 4096),
                    nn.ReLU(),
                    nn.BatchNorm1d(4096),
                    nn.Dropout(0.3),
                    nn.Linear(4096, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.2),
                    nn.Linear(512, 2)
                )

            # Freeze backbone parameters initially (only classifier trains)
            for name, p in net.named_parameters():
                if 'fc' in name or 'classifier' in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            net = net.to(device)

            # Train single backbone
            net, history = train_model(net, train_loader, val_loader, device, num_epochs=num_epochs, fine_tune_after=fine_tune_after)
            trained_models[backbone] = net
            histories[backbone] = history

            # Save per-backbone artifacts
            save_model_and_results(net, history, f"{model_name}_{backbone}", save_dir)

        # Ensemble evaluation on test set: majority voting with tie-break on averaged probabilities
        print("\n[INFO] Evaluating ensemble by majority voting (tie-break: average probs)...")
        import numpy as _np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        # Collect per-backbone probs for the entire test set
        all_probs = {bk: [] for bk in trained_models.keys()}
        y_true = []
        for data, target in test_loader:
            data = data.to(device)
            for backbone, model_obj in trained_models.items():
                model_obj.eval()
                with torch.no_grad():
                    out = model_obj(data)
                    probs = nn.functional.softmax(out, dim=1).cpu().numpy()
                all_probs[backbone].append(probs)
            y_true.extend(target.numpy().tolist())

        # Stack into arrays (n_models, N, C)
        probs_list = []
        for bk in trained_models.keys():
            probs_list.append(_np.vstack(all_probs[bk]))

        N = probs_list[0].shape[0]
        num_classes = probs_list[0].shape[1]

        # Majority voting with tie-break
        model_preds = [p.argmax(axis=1) for p in probs_list]  # list of arrays (N,)
        model_preds = _np.stack(model_preds, axis=0)  # (n_models, N)

        final_preds = []
        for i in range(N):
            votes = model_preds[:, i]
            vals, counts = _np.unique(votes, return_counts=True)
            max_count = counts.max()
            winners = vals[counts == max_count]
            if len(winners) == 1:
                final = int(winners[0])
            else:
                # tie: use average probability across models
                avg_prob = _np.mean(_np.stack([p[i] for p in probs_list], axis=0), axis=0)
                final = int(_np.argmax(avg_prob))
            final_preds.append(final)

        y_true = _np.array(y_true)
        final_preds = _np.array(final_preds)

        ensemble_metrics = {
            'accuracy': float(accuracy_score(y_true, final_preds)),
            'precision': float(precision_score(y_true, final_preds, zero_division=0)),
            'recall': float(recall_score(y_true, final_preds, zero_division=0)),
            'f1': float(f1_score(y_true, final_preds, zero_division=0))
        }
        # Save ensemble summary
        final_results = {
            'model_name': model_name,
            'backbones': backbones,
            'ensemble_method': 'majority_voting',
            'ensemble_metrics': ensemble_metrics,
            'timestamp': datetime.now().isoformat(),
            'histories': histories
        }
        summary_file = os.path.join(save_dir, f"{model_name}_ensemble_results.json")
        with open(summary_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n[OK] Ensemble evaluation complete. Results saved to: {summary_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up GPU memory
        clear_gpu_memory()

if __name__ == "__main__":
    main()