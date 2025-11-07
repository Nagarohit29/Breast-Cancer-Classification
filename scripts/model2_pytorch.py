"""
Model 2: ResNet-50 Transfer Learning for Breast Cancer Classification
Uses pre-trained ResNet-50 with transfer learning and fine-tuning
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
import sys

# Add project root to Python path so src imports work
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.utils.path_resolver import resolve_breakhis_path
from tqdm import tqdm
import random

# Import custom utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.gpu_utils import setup_device, clear_gpu_memory, set_reproducible_training
from src.utils.evaluation_metrics import ModelEvaluator, print_evaluation_summary
from src.utils.confusion_matrix import ConfusionMatrixGenerator
from src.utils.auc_display import display_auc_results

class TransferLearningCNN(nn.Module):
    """ResNet-50 based transfer learning model"""
    
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(TransferLearningCNN, self).__init__()
        
        # Load pre-trained ResNet-50 (try modern API first, then fallbacks)
        self.pretrained = False
        try:
            # torchvision >=0.13: preferred weights API
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1
            self.backbone = models.resnet50(weights=weights)
            self.pretrained = True
        except Exception:
            try:
                # Older torchvision API
                self.backbone = models.resnet50(pretrained=True)
                self.pretrained = True
            except Exception as e:
                # Offline or unable to download weights — fall back to random init
                print(f"⚠️  Warning: Could not load pretrained ResNet-50 weights ({e}). Using randomly initialized backbone.")
                # Create backbone without pretrained weights
                try:
                    self.backbone = models.resnet50(weights=None)  # torchvision newer API
                except Exception:
                    self.backbone = models.resnet50(pretrained=False)  # older API
                self.pretrained = False
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        
        # Custom classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.freeze_backbone = freeze_backbone
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("🔓 Backbone unfrozen for fine-tuning")
    
    def forward(self, x):
        return self.backbone(x)

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
            LR = float(os.getenv('LR', 0.01))  # Updated default learning rate
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
        # This method is superseded by resolve_breakhis_path; keep for compatibility
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


def create_data_loaders(batch_size=32, data_dir=None, magnification="100X"):
    """Create data loaders with enhanced augmentation for transfer learning"""
    
    # Enhanced train transforms for transfer learning
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
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
        print(f"⚠️  Warning: CSV files not found: {e}")
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, num_epochs=25, fine_tune_after=15, lr=1e-3, optim_name='sgd'):
    """Train the ResNet-50 model with optional fine-tuning"""
    
    # Loss function with class weights (if needed)
    criterion = nn.CrossEntropyLoss()
    
    # Different optimizers for different phases (use SGD w/ momentum by default)
    try:
        # prefer SGD for paper-like training
        optimizer = optim.SGD(model.backbone.fc.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    except Exception:
        optimizer = optim.Adam(model.backbone.fc.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    fine_tuning_started = False
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"🔒 Phase 1: Feature extraction (frozen backbone) - Epochs 1-{fine_tune_after}")
    print(f"🔓 Phase 2: Fine-tuning (unfrozen backbone) - Epochs {fine_tune_after+1}-{num_epochs}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Switch to fine-tuning phase
        if epoch == fine_tune_after and model.freeze_backbone:
            print(f"\n🔄 Switching to fine-tuning phase...")
            model.unfreeze_backbone()
            # Use smaller learning rate for fine-tuning; switch to Adam for stability
            optimizer = optim.Adam(model.parameters(), lr=lr * 0.1, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            fine_tuning_started = True
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{num_epochs}', unit='batch')):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            running_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == target.data)
            total_samples += data.size(0)

            if batch_idx % 50 == 0:
                phase = "Fine-tuning" if fine_tuning_started else "Feature extraction"
                tqdm.write(f'   Epoch {epoch+1}/{num_epochs} [{phase}], Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_loss = running_loss / total_samples
        train_acc = running_corrects.double() / total_samples
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_running_loss += loss.item() * data.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == target.data)
                val_total_samples += data.size(0)
        
        val_loss = val_running_loss / val_total_samples
        val_acc = val_running_corrects.double() / val_total_samples
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Record history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['learning_rate'].append(float(current_lr))
        
        epoch_time = time.time() - epoch_start
        phase = "Fine-tuning" if fine_tuning_started else "Feature extraction"
        print(f'Epoch {epoch+1}/{num_epochs} [{phase}] completed in {epoch_time:.1f}s')
        print(f'   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'   Learning Rate: {current_lr:.6f}')
        print(f'   Best Val Acc: {best_val_acc:.4f}')
        print('-' * 60)
    
    total_time = time.time() - start_time
    print(f'✅ Training completed in {total_time:.1f}s')
    print(f'🏆 Best validation accuracy: {best_val_acc:.4f}')
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history

def save_model_and_results(model, history, model_name, save_dir):
    """Save model, training history, and create results directory"""
    
    # Create directories
    model_dir = os.path.join("models", model_name)
    results_dir = os.path.join("results", model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history_path = os.path.join(save_dir, f"{model_name}_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Also save to results directory
    results_history_path = os.path.join(results_dir, f"{model_name}_training_history.json")
    with open(results_history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Training history saved to: {history_path}")
    print(f"✅ Results copied to: {results_dir}")

def main():
    """Main training function for Model 2"""
    
    print("="*80)
    print("🔬 MODEL 2: ResNet-50 Transfer Learning for Breast Cancer Classification")
    print("="*80)
    
    # Config (env overrides for paper matching)
    SEED = int(os.getenv('SEED', 42))
    TRAIN_EPOCHS = int(os.getenv('TRAIN_EPOCHS', 50))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    LR = float(os.getenv('LR', 1e-3))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 1e-4))
    OPTIM = os.getenv('OPTIM', 'adam')

    # Deterministic seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # also call helper if available
    try:
        set_reproducible_training(SEED)
    except Exception:
        pass
    
    # Setup device
    device, device_info = setup_device()
    
    # Model parameters
    model_name = "model2_resnet50_transfer"
    batch_size = BATCH_SIZE
    num_epochs = TRAIN_EPOCHS
    fine_tune_after = 15
    
    # Create save directories
    save_dir = os.path.join("results", model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Read optional dataset overrides
        data_dir = os.getenv('DATA_DIR', None)
        magnification = os.getenv('MAGNIFICATION', 'all_mags')

        # Create data loaders
        print("\n📁 Loading data...")
        train_loader, val_loader, test_loader = create_data_loaders(batch_size=batch_size, data_dir=data_dir, magnification=magnification)
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")

        # Create model
        print(f"\n🏗️  Creating {model_name}...")
        model = TransferLearningCNN(num_classes=2, freeze_backbone=True).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters (initial): {trainable_params:,}")
        print(f"   Device: {device_info}")
        print(f"   Transfer learning: ResNet-50 backbone")
        print(f"   Fine-tuning: After epoch {fine_tune_after}")

        # Train model
        model, history = train_model(model, train_loader, val_loader, device, num_epochs, fine_tune_after, lr=LR, optim_name=OPTIM)

        # Save model and results
        save_model_and_results(model, history, model_name, save_dir)

        # Comprehensive evaluation
        print(f"\n📊 Starting comprehensive evaluation...")
        evaluator = ModelEvaluator(model_name, device)
        metrics = evaluator.evaluate_model(model, test_loader, save_dir)
        print_evaluation_summary(metrics)

        # Display AUC prominently
        display_auc_results(metrics, model_name)

        # Generate confusion matrix
        print(f"\n🔍 Generating confusion matrix...")
        cm_generator = ConfusionMatrixGenerator(model_name)
        cm_analysis = cm_generator.generate_confusion_matrix(model, test_loader, device, save_dir)

        # Save final summary
        final_results = {
            'model_name': model_name,
            'model_architecture': 'ResNet-50 Transfer Learning',
            'backbone': 'ResNet-50 (pre-trained on ImageNet)',
            'total_parameters': total_params,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'training_device': device_info,
            'training_epochs': num_epochs,
            'fine_tune_after_epoch': fine_tune_after,
            'batch_size': batch_size,
            'training_history': history,
            'evaluation_metrics': metrics,
            'confusion_matrix_analysis': cm_analysis,
            'timestamp': datetime.now().isoformat()
        }

        summary_file = os.path.join(save_dir, f"{model_name}_complete_results.json")
        with open(summary_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n✅ Model 2 training and evaluation completed successfully!")
        print(f"📁 All results saved to: {save_dir}")
        print(f"💾 Complete summary: {summary_file}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up GPU memory
        clear_gpu_memory()

if __name__ == "__main__":
    main()