"""
Feature Extraction for Traditional ML Models
Extracts handcrafted features (LBP, Gabor, GLCM) from breast cancer histopathology images.
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from datetime import datetime
import os

# Paths
PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = PROCESSED_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Parameters for feature extraction
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

def extract_lbp_features(gray_img):
    """
    Extract Local Binary Pattern histogram features.
    
    Args:
        gray_img (np.ndarray): Grayscale image
        
    Returns:
        np.ndarray: LBP histogram features
    """
    lbp = local_binary_pattern(gray_img, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, LBP_POINTS + 3),
                           range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # normalize
    return hist

def extract_gabor_features(gray_img):
    """
    Extract mean and variance of Gabor filter responses.
    
    Args:
        gray_img (np.ndarray): Grayscale image
        
    Returns:
        np.ndarray: Gabor filter features
    """
    feats = []
    for theta in (0, np.pi/4, np.pi/2, 3*np.pi/4):  # 4 orientations
        filt_real, filt_imag = gabor(gray_img, frequency=0.6, theta=theta)
        feats.append(filt_real.mean())
        feats.append(filt_real.var())
    return np.array(feats)

def extract_glcm_features(gray_img):
    """
    Extract GLCM (Gray-Level Co-occurrence Matrix) texture features.
    
    Args:
        gray_img (np.ndarray): Grayscale image
        
    Returns:
        np.ndarray: GLCM texture features
    """
    # Ensure image is in proper range and type
    if gray_img.dtype != np.uint8:
        gray_img = (gray_img * 255).astype(np.uint8)
    
    # Reduce levels for computational efficiency
    gray_img = (gray_img / 4).astype(np.uint8)  # 64 levels instead of 256
    
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)
    
    # Extract texture properties
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm])

def extract_additional_features(gray_img):
    """
    Extract additional statistical and morphological features.
    
    Args:
        gray_img (np.ndarray): Grayscale image
        
    Returns:
        np.ndarray: Additional features
    """
    features = []
    
    # Statistical features
    features.append(gray_img.mean())
    features.append(gray_img.std())
    features.append(np.median(gray_img))
    features.append(gray_img.min())
    features.append(gray_img.max())
    features.append(np.percentile(gray_img, 25))
    features.append(np.percentile(gray_img, 75))
    
    # Histogram features
    hist, _ = np.histogram(gray_img, bins=32, range=(0, 256))
    hist = hist.astype(float) / (hist.sum() + 1e-6)
    
    # Entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    features.append(entropy)
    
    # Skewness and Kurtosis approximation
    mean_val = gray_img.mean()
    std_val = gray_img.std()
    if std_val > 0:
        skewness = np.mean(((gray_img - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((gray_img - mean_val) / std_val) ** 4)
    else:
        skewness = 0
        kurtosis = 0
    
    features.append(skewness)
    features.append(kurtosis)
    
    return np.array(features)

def process_split(split_name):
    """
    Process a single split (train/test/deploy) to extract features.
    
    Args:
        split_name (str): Name of the split
        
    Returns:
        dict: Statistics about processed images
    """
    split_path = PROCESSED_DIR / split_name
    features = []
    labels = []
    stats = {"benign": 0, "malignant": 0, "errors": 0}
    
    if not split_path.exists():
        print(f"⚠️  Warning: {split_path} does not exist. Skipping {split_name} split.")
        return stats

    print(f"📂 Processing {split_name.upper()} split for feature extraction...")
    
    for label in ["benign", "malignant"]:
        class_dir = split_path / label
        
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_dir} does not exist. Skipping {label} class.")
            continue
        
        # Find all image files
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
        images = []
        for ext in image_extensions:
            images.extend(list(class_dir.glob(ext)))
        
        if not images:
            print(f"⚠️  No images found in {class_dir}")
            continue
        
        processed_count = 0
        for img_path in tqdm(images, desc=f"Extracting features {split_name}/{label}"):
            try:
                # Load image in grayscale
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    stats["errors"] += 1
                    continue
                
                # Resize to standard size for consistent features
                img = cv2.resize(img, (224, 224))
                
                # Extract different types of features
                lbp_feats = extract_lbp_features(img)
                gabor_feats = extract_gabor_features(img)
                glcm_feats = extract_glcm_features(img)
                additional_feats = extract_additional_features(img)
                
                # Concatenate all features
                feature_vector = np.hstack([lbp_feats, gabor_feats, glcm_feats, additional_feats])
                
                features.append(feature_vector)
                labels.append(label)
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                stats["errors"] += 1
                continue
        
        stats[label] = processed_count
        print(f"✅ Extracted features from {processed_count} {label} images")

    # Save features as CSV
    if features:
        feature_names = []
        
        # LBP feature names
        feature_names.extend([f"lbp_{i}" for i in range(LBP_POINTS + 2)])
        
        # Gabor feature names
        for theta_idx, theta in enumerate([0, 45, 90, 135]):
            feature_names.extend([f"gabor_mean_{theta}", f"gabor_var_{theta}"])
        
        # GLCM feature names
        feature_names.extend(["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", 
                             "glcm_energy", "glcm_correlation", "glcm_asm"])
        
        # Additional feature names
        feature_names.extend(["mean", "std", "median", "min", "max", "q25", "q75", 
                             "entropy", "skewness", "kurtosis"])
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=feature_names)
        df["label"] = labels
        df["filename"] = [f"{label}_{i}" for i, label in enumerate(labels)]
        
        # Save to CSV
        out_file = FEATURES_DIR / f"{split_name}_features.csv"
        df.to_csv(out_file, index=False)
        
        print(f"💾 Saved {len(features)} feature vectors to {out_file}")
        print(f"📊 Feature vector size: {len(feature_names)} features")
    else:
        print(f"⚠️  No features extracted for {split_name} split")
    
    return stats

def create_feature_info():
    """Create a JSON file with information about extracted features."""
    feature_info = {
        "extraction_date": datetime.now().isoformat(),
        "feature_types": {
            "LBP": {
                "description": "Local Binary Pattern histogram",
                "parameters": {
                    "radius": LBP_RADIUS,
                    "points": LBP_POINTS,
                    "method": LBP_METHOD
                },
                "feature_count": LBP_POINTS + 2
            },
            "Gabor": {
                "description": "Gabor filter responses (mean and variance)",
                "parameters": {
                    "frequency": 0.6,
                    "orientations": [0, 45, 90, 135]
                },
                "feature_count": 8  # 4 orientations × 2 (mean + var)
            },
            "GLCM": {
                "description": "Gray-Level Co-occurrence Matrix features",
                "parameters": {
                    "distances": [1],
                    "angles": [0],
                    "levels": 64
                },
                "features": ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"],
                "feature_count": 6
            },
            "Statistical": {
                "description": "Additional statistical and morphological features",
                "features": ["mean", "std", "median", "min", "max", "q25", "q75", "entropy", "skewness", "kurtosis"],
                "feature_count": 10
            }
        },
        "total_features": LBP_POINTS + 2 + 8 + 6 + 10,
        "image_preprocessing": {
            "resize": [224, 224],
            "color_space": "grayscale"
        }
    }
    
    info_file = FEATURES_DIR / "feature_info.json"
    with open(info_file, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"📋 Feature information saved to {info_file}")

def run_feature_extraction():
    """
    Run feature extraction for all splits (train, test, deploy).
    """
    print("🔍 Starting Handcrafted Feature Extraction")
    print("=" * 60)
    print(f"📊 Features to extract:")
    print(f"   • LBP: {LBP_POINTS + 2} features")
    print(f"   • Gabor: 8 features (4 orientations × 2 stats)")
    print(f"   • GLCM: 6 texture features")
    print(f"   • Statistical: 10 additional features")
    print(f"   • Total: {LBP_POINTS + 2 + 8 + 6 + 10} features per image")
    print("=" * 60)
    
    # Check if processed directory exists
    if not PROCESSED_DIR.exists():
        print(f"❌ Error: {PROCESSED_DIR} does not exist.")
        print("Please ensure images are preprocessed and split first.")
        return
    
    # Process each split
    all_stats = {}
    splits = ["train", "test", "deploy"]
    
    for split_name in splits:
        stats = process_split(split_name)
        all_stats[split_name] = stats
    
    # Create feature information file
    create_feature_info()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 FEATURE EXTRACTION COMPLETE!")
    print("=" * 60)
    
    total_processed = 0
    total_errors = 0
    
    for split_name, stats in all_stats.items():
        split_total = stats["benign"] + stats["malignant"]
        total_processed += split_total
        total_errors += stats["errors"]
        
        if split_total > 0:
            print(f"📊 {split_name.capitalize()}: {split_total} images "
                  f"({stats['benign']} benign, {stats['malignant']} malignant)")
        else:
            print(f"⚠️  {split_name.capitalize()}: No features extracted")
    
    print(f"\n📈 Total processed: {total_processed} images")
    if total_errors > 0:
        print(f"⚠️  Errors encountered: {total_errors} images")
    
    print(f"💾 Features saved in: {FEATURES_DIR}")
    print("🚀 Ready for traditional ML model training!")

def check_feature_extraction_status():
    """
    Check if feature extraction has been completed.
    
    Returns:
        bool: True if feature extraction is complete, False otherwise
    """
    required_files = ["train_features.csv", "test_features.csv", "deploy_features.csv"]
    
    for filename in required_files:
        if not (FEATURES_DIR / filename).exists():
            return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract handcrafted features from breast cancer images')
    parser.add_argument('--check', action='store_true',
                       help='Check if feature extraction is already completed')
    parser.add_argument('--force', action='store_true',
                       help='Force re-extraction even if already completed')
    
    args = parser.parse_args()
    
    if args.check:
        is_completed = check_feature_extraction_status()
        if is_completed:
            print("✅ Feature extraction is already completed!")
            print(f"📁 Features available in: {FEATURES_DIR}")
        else:
            print("❌ Feature extraction not yet completed.")
        exit(0)
    
    # Check if already completed (unless forced)
    if not args.force and check_feature_extraction_status():
        print("✅ Feature extraction already completed!")
        print("Use --force to re-extract or --check to verify status.")
        exit(0)
    
    run_feature_extraction()