"""
Preprocessing Script for BreaKHis Breast Cancer Dataset
Handles image preprocessing, augmentation, and tensor conversion for training.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import json
from datetime import datetime

# Paths
PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = PROCESSED_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Image size for models like ResNet50, VGG16
IMG_SIZE = 224  

# Normalization (ImageNet mean & std)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Transform pipelines
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

def preprocess_split(split_name, transform):
    """
    Preprocess images in a specific split (train/test/deploy).
    
    Args:
        split_name (str): Name of the split ('train', 'test', or 'deploy')
        transform: PyTorch transform to apply
    """
    split_path = PROCESSED_DIR / split_name
    split_stats = {"benign": 0, "malignant": 0}
    
    if not split_path.exists():
        print(f"⚠️  Warning: {split_path} does not exist. Skipping {split_name} split.")
        return split_stats
    
    for label in ["benign", "malignant"]:
        class_dir = split_path / label
        
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_dir} does not exist. Skipping {label} class in {split_name}.")
            continue
            
        # Find all image files
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
        images = []
        for ext in image_extensions:
            images.extend(list(class_dir.glob(ext)))
        
        if not images:
            print(f"⚠️  No images found in {class_dir}")
            continue
            
        print(f"📁 Processing {len(images)} images in {split_name}/{label}")
        
        processed_count = 0
        for img_path in tqdm(images, desc=f"Processing {split_name}/{label}"):
            try:
                # Load and convert image
                img = Image.open(img_path).convert("RGB")
                
                # Apply transforms
                tensor = transform(img)
                
                # Save as .npy file (overwrite with preprocessed version)
                npy_path = class_dir / (img_path.stem + ".npy")
                np.save(npy_path, tensor.numpy())
                
                img.close()
                processed_count += 1
                
                # Optionally: remove original image if only npy required
                # Uncomment the line below if you want to keep only .npy files
                # os.remove(img_path)
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                continue
        
        split_stats[label] = processed_count
        print(f"✅ Processed {processed_count} {label} images in {split_name}")
    
    return split_stats

def update_splits_info(preprocessing_stats):
    """
    Update the splits_info.json file with preprocessing information.
    
    Args:
        preprocessing_stats (dict): Statistics from preprocessing
    """
    splits_info_path = Path("data/splits_info.json")
    
    try:
        # Load existing info or create new
        if splits_info_path.exists():
            with open(splits_info_path, 'r') as f:
                splits_info = json.load(f)
        else:
            splits_info = {}
        
        # Update preprocessing info
        splits_info.update({
            "preprocessing": {
                "completed": True,
                "completion_date": datetime.now().isoformat(),
                "image_size": IMG_SIZE,
                "normalization": {
                    "mean": imagenet_mean,
                    "std": imagenet_std
                },
                "augmentation_applied": True,
                "output_format": "numpy_tensors"
            },
            "processed_counts": preprocessing_stats
        })
        
        # Save updated info
        with open(splits_info_path, 'w') as f:
            json.dump(splits_info, f, indent=2)
            
        print(f"📊 Updated splits information in {splits_info_path}")
        
    except Exception as e:
        print(f"⚠️  Could not update splits_info.json: {e}")

def run_preprocessing():
    """
    Run preprocessing for all splits (train, test, deploy).
    """
    print("🔄 Starting BreaKHis Dataset Preprocessing")
    print("=" * 50)
    
    # Check if processed directory exists
    if not PROCESSED_DIR.exists():
        print(f"❌ Error: {PROCESSED_DIR} does not exist.")
        print("Please ensure the data has been split into train/test/deploy first.")
        return
    
    # Preprocessing statistics
    all_stats = {}
    
    # Process each split
    splits = [
        ("train", train_transform),
        ("test", test_transform),
        ("deploy", test_transform)
    ]
    
    for split_name, transform in splits:
        print(f"\n📂 Processing {split_name.upper()} split...")
        stats = preprocess_split(split_name, transform)
        all_stats[split_name] = stats
        
        total_images = sum(stats.values())
        print(f"✅ {split_name.capitalize()} split complete: {total_images} images processed")
    
    # Update splits info
    update_splits_info(all_stats)
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 PREPROCESSING COMPLETE!")
    print("=" * 50)
    
    total_processed = 0
    for split_name, stats in all_stats.items():
        split_total = sum(stats.values())
        total_processed += split_total
        print(f"📊 {split_name.capitalize()}: {split_total} images ({stats['benign']} benign, {stats['malignant']} malignant)")
    
    print(f"📈 Total processed: {total_processed} images")
    print(f"💾 Preprocessed tensors saved as .npy files")
    print(f"🖼️  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"📏 Normalization: ImageNet statistics")
    print("🚀 Ready for model training!")

def check_preprocessing_status():
    """
    Check if preprocessing has been completed.
    
    Returns:
        bool: True if preprocessing is complete, False otherwise
    """
    splits_info_path = Path("data/splits_info.json")
    
    if not splits_info_path.exists():
        return False
    
    try:
        with open(splits_info_path, 'r') as f:
            splits_info = json.load(f)
        
        return splits_info.get("preprocessing", {}).get("completed", False)
    
    except Exception:
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess BreaKHis dataset images')
    parser.add_argument('--check', action='store_true', 
                       help='Check if preprocessing is already completed')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if already completed')
    
    args = parser.parse_args()
    
    if args.check:
        is_completed = check_preprocessing_status()
        if is_completed:
            print("✅ Preprocessing is already completed!")
        else:
            print("❌ Preprocessing not yet completed.")
        exit(0)
    
    # Check if already completed (unless forced)
    if not args.force and check_preprocessing_status():
        print("✅ Preprocessing already completed!")
        print("Use --force to reprocess or --check to verify status.")
        exit(0)
    
    run_preprocessing()