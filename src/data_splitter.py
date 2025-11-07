"""
Data Splitter for BreaKHis Dataset
Splits raw breast cancer histopathology images into train/test/deploy sets (80/10/10).
Maintains class balance and creates proper directory structure.
"""

import os
import shutil
import random
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm

# Paths
RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Split ratios
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
DEPLOY_RATIO = 0.1
SEED = 42

def validate_ratios():
    """Validate that split ratios sum to 1.0"""
    total_ratio = TRAIN_RATIO + TEST_RATIO + DEPLOY_RATIO
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

def get_image_files(directory):
    """
    Get all image files from a directory.
    
    Args:
        directory (Path): Directory to search
        
    Returns:
        list: List of image file paths
    """
    if not directory.exists():
        return []
    
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    images = []
    
    for ext in image_extensions:
        images.extend(list(directory.glob(ext)))
    
    return images

def copy_files_with_progress(file_list, destination_dir, description):
    """
    Copy files with progress bar.
    
    Args:
        file_list (list): List of file paths to copy
        destination_dir (Path): Destination directory
        description (str): Description for progress bar
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in tqdm(file_list, desc=description):
        try:
            dest_path = destination_dir / file_path.name
            shutil.copy2(file_path, dest_path)
        except Exception as e:
            print(f"⚠️  Error copying {file_path}: {e}")

def split_data():
    """
    Split the raw dataset into train/test/deploy sets.
    """
    print("🔄 Starting BreaKHis Dataset Splitting")
    print("=" * 60)
    print(f"📊 Split ratios: Train {TRAIN_RATIO*100}%, Test {TEST_RATIO*100}%, Deploy {DEPLOY_RATIO*100}%")
    print(f"🎲 Random seed: {SEED}")
    print("=" * 60)
    
    # Validate inputs
    validate_ratios()
    random.seed(SEED)
    
    if not RAW_DATA_DIR.exists():
        print(f"❌ Error: {RAW_DATA_DIR} does not exist.")
        print("Please place your raw BreaKHis images in the data/raw directory first.")
        return
    
    split_info = {
        "dataset": "BreaKHis",
        "split_strategy": "stratified",
        "split_ratios": {
            "train": TRAIN_RATIO,
            "test": TEST_RATIO,
            "deploy": DEPLOY_RATIO
        },
        "random_seed": SEED,
        "split_date": datetime.now().isoformat(),
        "class_counts": {},
        "total_images": 0
    }
    
    total_images_processed = 0
    
    for label in ["benign", "malignant"]:
        print(f"\n📂 Processing {label.upper()} class...")
        
        input_dir = RAW_DATA_DIR / label
        images = get_image_files(input_dir)
        
        if not images:
            print(f"⚠️  No images found in {input_dir}")
            split_info["class_counts"][label] = {
                "train": 0, "test": 0, "deploy": 0, "total": 0
            }
            continue
        
        print(f"📊 Found {len(images)} {label} images")
        
        # Shuffle images for randomness
        random.shuffle(images)
        
        # First split: separate training from (test + deploy)
        train_imgs, temp_imgs = train_test_split(
            images, 
            test_size=(1 - TRAIN_RATIO), 
            random_state=SEED,
            shuffle=True
        )
        
        # Second split: separate test from deploy
        if temp_imgs:  # Only split if there are images to split
            test_imgs, deploy_imgs = train_test_split(
                temp_imgs, 
                test_size=DEPLOY_RATIO / (TEST_RATIO + DEPLOY_RATIO),
                random_state=SEED,
                shuffle=True
            )
        else:
            test_imgs, deploy_imgs = [], []
        
        # Store splits
        splits = {
            "train": train_imgs,
            "test": test_imgs,
            "deploy": deploy_imgs
        }
        
        # Copy files to respective directories
        for split_name, img_list in splits.items():
            if img_list:
                split_dir = OUTPUT_DIR / split_name / label
                description = f"Copying {split_name}/{label}"
                copy_files_with_progress(img_list, split_dir, description)
                print(f"✅ Copied {len(img_list)} images to {split_dir}")
            else:
                print(f"⚠️  No images to copy for {split_name}/{label}")
        
        # Update split info
        class_total = len(images)
        split_info["class_counts"][label] = {
            "train": len(train_imgs),
            "test": len(test_imgs), 
            "deploy": len(deploy_imgs),
            "total": class_total
        }
        
        total_images_processed += class_total
        
        # Print class summary
        print(f"📈 {label.capitalize()} split summary:")
        print(f"   • Train: {len(train_imgs)} ({len(train_imgs)/class_total*100:.1f}%)")
        print(f"   • Test: {len(test_imgs)} ({len(test_imgs)/class_total*100:.1f}%)")
        print(f"   • Deploy: {len(deploy_imgs)} ({len(deploy_imgs)/class_total*100:.1f}%)")
    
    # Update total count
    split_info["total_images"] = total_images_processed
    
    # Add processing metadata
    split_info["preprocessing"] = {
        "resize_to": [224, 224],
        "normalization": "imagenet",
        "augmentation_applied": False
    }
    
    split_info["notes"] = "Images split maintaining class balance. Original files preserved in data/raw."
    
    # Save metadata to splits_info.json
    splits_info_path = OUTPUT_DIR.parent / "splits_info.json"
    with open(splits_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 DATA SPLITTING COMPLETE!")
    print("=" * 60)
    
    # Calculate totals across classes
    total_train = sum(split_info["class_counts"][cls]["train"] for cls in ["benign", "malignant"])
    total_test = sum(split_info["class_counts"][cls]["test"] for cls in ["benign", "malignant"])
    total_deploy = sum(split_info["class_counts"][cls]["deploy"] for cls in ["benign", "malignant"])
    
    print(f"📊 Final split summary:")
    print(f"   • Train: {total_train} images ({total_train/total_images_processed*100:.1f}%)")
    print(f"   • Test: {total_test} images ({total_test/total_images_processed*100:.1f}%)")
    print(f"   • Deploy: {total_deploy} images ({total_deploy/total_images_processed*100:.1f}%)")
    print(f"   • Total: {total_images_processed} images")
    
    print(f"\n📁 Split files saved in: {OUTPUT_DIR}")
    print(f"📋 Metadata saved in: {splits_info_path}")
    
    # Display detailed breakdown
    print(f"\n📈 Detailed class breakdown:")
    for label in ["benign", "malignant"]:
        counts = split_info["class_counts"][label]
        print(f"   {label.capitalize()}:")
        print(f"     - Train: {counts['train']}")
        print(f"     - Test: {counts['test']}")
        print(f"     - Deploy: {counts['deploy']}")
        print(f"     - Total: {counts['total']}")
    
    print("\n🚀 Ready for preprocessing!")
    print("Next steps:")
    print("  1. Run: python src/data/preprocessing.py (for deep learning)")
    print("  2. Run: python src/data/feature_extraction.py (for traditional ML)")

def check_split_status():
    """
    Check if data splitting has been completed.
    
    Returns:
        bool: True if splitting is complete, False otherwise
    """
    # Check if all required directories exist and have files
    required_dirs = [
        OUTPUT_DIR / "train" / "benign",
        OUTPUT_DIR / "train" / "malignant",
        OUTPUT_DIR / "test" / "benign",
        OUTPUT_DIR / "test" / "malignant",
        OUTPUT_DIR / "deploy" / "benign",
        OUTPUT_DIR / "deploy" / "malignant"
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            return False
        
        # Check if directory has any image files
        if not get_image_files(dir_path):
            return False
    
    # Check if splits_info.json exists
    splits_info_path = OUTPUT_DIR.parent / "splits_info.json"
    if not splits_info_path.exists():
        return False
    
    return True

def clean_split_directories():
    """
    Clean existing split directories before re-splitting.
    """
    split_dirs = ["train", "test", "deploy"]
    
    for split_dir in split_dirs:
        split_path = OUTPUT_DIR / split_dir
        if split_path.exists():
            print(f"🧹 Cleaning {split_path}")
            shutil.rmtree(split_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split BreaKHis dataset into train/test/deploy')
    parser.add_argument('--check', action='store_true',
                       help='Check if splitting is already completed')
    parser.add_argument('--force', action='store_true',
                       help='Force re-splitting even if already completed')
    parser.add_argument('--clean', action='store_true',
                       help='Clean existing split directories first')
    
    args = parser.parse_args()
    
    if args.check:
        is_completed = check_split_status()
        if is_completed:
            print("✅ Data splitting is already completed!")
            print(f"📁 Split data available in: {OUTPUT_DIR}")
        else:
            print("❌ Data splitting not yet completed.")
            print("Run: python src/data_splitter.py")
        exit(0)
    
    if args.clean:
        clean_split_directories()
        print("🧹 Cleaned existing split directories")
    
    # Check if already completed (unless forced)
    if not args.force and not args.clean and check_split_status():
        print("✅ Data splitting already completed!")
        print("Use --force to re-split or --check to verify status.")
        print("Use --clean to remove existing splits first.")
        exit(0)
    
    split_data()