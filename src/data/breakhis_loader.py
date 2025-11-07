"""
Data Loader for BreaKHis 80-10-10 splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import json

class BreaKHisDataLoader:
    """
    Data loader for BreaKHis dataset with 80-10-10 splits
    """
    
    def __init__(self, data_dir="data/processed", base_image_dir="data/raw"):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing the split CSV files
            base_image_dir: Base directory containing the actual image files (now local)
        """
        self.data_dir = Path(data_dir)
        self.base_image_dir = Path(base_image_dir)
        self.label_encoder = LabelEncoder()
        
    def load_split_data(self, magnification=None, return_paths=True):
        """
        Load train, validation, and test splits
        
        Args:
            magnification: Specific magnification (40, 100, 200, 400) or None for all
            return_paths: If True, return file paths; if False, return loaded images
            
        Returns:
            tuple: (train_data, val_data, test_data, metadata)
        """
        
        # Construct file suffix
        if magnification == 'all_mags':
            mag_suffix = "_all_mags"
        elif magnification:
            mag_suffix = f"_{magnification}X"
        else:
            mag_suffix = "_all_mags"
        
        # Load CSV files
        train_file = self.data_dir / f"train{mag_suffix}.csv"
        val_file = self.data_dir / f"validation{mag_suffix}.csv"
        test_file = self.data_dir / f"test{mag_suffix}.csv"
        metadata_file = self.data_dir / f"split_metadata{mag_suffix}.json"
        
        # Check if files exist
        for file in [train_file, val_file, test_file, metadata_file]:
            if not file.exists():
                raise FileNotFoundError(f"Split file not found: {file}")
        
        # Load dataframes
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract class labels and create full paths
        for df in [train_df, val_df, test_df]:
            df['class'] = df['filename'].apply(self._extract_class_from_path)
            df['full_path'] = df['filename'].apply(self._map_csv_path_to_actual_path)
        
        # Encode labels
        all_labels = pd.concat([train_df['class'], val_df['class'], test_df['class']])
        self.label_encoder.fit(all_labels)
        
        train_df['label'] = self.label_encoder.transform(train_df['class'])
        val_df['label'] = self.label_encoder.transform(val_df['class'])
        test_df['label'] = self.label_encoder.transform(test_df['class'])
        
        if return_paths:
            return (train_df, val_df, test_df, metadata)
        else:
            # TODO: Implement image loading if needed
            raise NotImplementedError("Image loading not implemented yet")
    
    def get_data_for_training(self, magnification=None):
        """
        Get data in format ready for training
        
        Args:
            magnification: Specific magnification or None for all
            
        Returns:
            dict: Dictionary with train/val/test data and labels
        """
        train_df, val_df, test_df, metadata = self.load_split_data(magnification)
        
        return {
            'train': {
                'paths': train_df['full_path'].values,
                'labels': train_df['label'].values,
                'classes': train_df['class'].values,
                'df': train_df
            },
            'validation': {
                'paths': val_df['full_path'].values,
                'labels': val_df['label'].values,
                'classes': val_df['class'].values,
                'df': val_df
            },
            'test': {
                'paths': test_df['full_path'].values,
                'labels': test_df['label'].values,
                'classes': test_df['class'].values,
                'df': test_df
            },
            'metadata': metadata,
            'label_encoder': self.label_encoder,
            'class_names': self.label_encoder.classes_
        }
    
    def load_split(self, split_name, magnification=None):
        """
        Load a specific split (train, validation, test)
        
        Args:
            split_name: 'train', 'validation', or 'test'
            magnification: Specific magnification or None for all
            
        Returns:
            pandas.DataFrame: DataFrame with image paths and labels
        """
        train_df, val_df, test_df, metadata = self.load_split_data(magnification)
        
        if split_name == 'train':
            df = train_df.copy()
        elif split_name == 'validation':
            df = val_df.copy()
        elif split_name == 'test':
            df = test_df.copy()
        else:
            raise ValueError(f"Invalid split_name: {split_name}. Must be 'train', 'validation', or 'test'")
        
        # Add image_path column for compatibility
        df['image_path'] = df['full_path']
        
        return df
    
    def _extract_class_from_path(self, filepath):
        """Extract class label from BreaKHis file path"""
        if '/benign/' in filepath:
            return 'benign'
        elif '/malignant/' in filepath:
            return 'malignant'
        else:
            return 'unknown'
    
    def _map_csv_path_to_actual_path(self, csv_path):
        """
        Map CSV file path to actual file system path
        
        CSV path: BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/100X/SOB_B_A-14-22549AB-100-001.png
        Actual path: E:/ML project/breast-cancer-classification/data/raw/benign/SOB/adenosis/SOB_B_A_14-22549AB/100X/SOB_B_A-14-22549AB-100-001.png
        """
        # Remove the BreaKHis_v1/histology_slides/breast/ prefix
        if 'BreaKHis_v1/histology_slides/breast/' in csv_path:
            relative_path = csv_path.replace('BreaKHis_v1/histology_slides/breast/', '')
            # Convert to absolute path - go up to project root (breast-cancer-classification)
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent  # Go up 3 levels to breast-cancer-classification
            actual_path = project_root / self.base_image_dir / relative_path
            return str(actual_path)
        else:
            # Fallback: assume it's already a relative path
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent  # Go up 3 levels to breast-cancer-classification
            actual_path = project_root / self.base_image_dir / csv_path
            return str(actual_path)
    
    def print_split_summary(self, magnification=None):
        """Print summary of the data splits"""
        _, _, _, metadata = self.load_split_data(magnification)
        
        print(f"=== BreaKHis Data Split Summary ===")
        if magnification:
            print(f"Magnification: {magnification}X")
        else:
            print("Magnification: All")
        
        print(f"Total samples: {metadata['total_samples']}")
        print(f"Total patients: {metadata['total_patients']}")
        
        print(f"\nSplit Distribution:")
        for split_name, split_info in metadata['splits'].items():
            print(f"  {split_name.title()}: {split_info['samples']} samples "
                  f"({split_info['patients']} patients, {split_info['percentage']:.1f}%)")
        
        print(f"\nClass Distribution:")
        for split_name, class_dist in metadata['class_distribution'].items():
            if split_name == 'overall':
                continue
            print(f"  {split_name.title()}:")
            total_split = sum(class_dist.values())
            for class_name, count in class_dist.items():
                percentage = count / total_split * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")

# Example usage functions
def load_data_for_sklearn(magnification=100):
    """
    Load data in format suitable for sklearn models
    
    Args:
        magnification: Magnification level (40, 100, 200, 400)
    
    Returns:
        tuple: (X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test)
    """
    loader = BreaKHisDataLoader()
    data = loader.get_data_for_training(magnification)
    
    return (
        data['train']['paths'],
        data['validation']['paths'], 
        data['test']['paths'],
        data['train']['labels'],
        data['validation']['labels'],
        data['test']['labels']
    )

def load_data_for_pytorch(magnification=100):
    """
    Load data in format suitable for PyTorch models
    
    Args:
        magnification: Magnification level (40, 100, 200, 400)
    
    Returns:
        dict: Data dictionary with all necessary information
    """
    loader = BreaKHisDataLoader()
    return loader.get_data_for_training(magnification)

if __name__ == "__main__":
    # Example usage
    print("=== BreaKHis Data Loader Demo ===\n")
    
    loader = BreaKHisDataLoader()
    
    # Print summaries for different magnifications
    for mag in [None, 100, 200]:
        loader.print_split_summary(mag)
        print()
    
    # Load data for training
    print("Loading data for 100X magnification...")
    data = load_data_for_pytorch(100)
    
    print(f"Training samples: {len(data['train']['paths'])}")
    print(f"Validation samples: {len(data['validation']['paths'])}")
    print(f"Test samples: {len(data['test']['paths'])}")
    print(f"Classes: {data['class_names']}")