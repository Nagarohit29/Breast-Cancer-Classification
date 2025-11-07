"""
Configuration management for the breast cancer classification project.
"""

import yaml
import json
from pathlib import Path

class Config:
    """Configuration class for managing experiment parameters."""
    
    def __init__(self, config_dict=None):
        """
        Initialize configuration.
        
        Args:
            config_dict (dict): Configuration dictionary
        """
        if config_dict is None:
            config_dict = {}
            
        self.config = config_dict
        
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
        
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def update(self, other_config):
        self.config.update(other_config)
        
    def to_dict(self):
        return self.config
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path (str): Path to YAML configuration file
            
        Returns:
            Config: Configuration object
        """
        with open(yaml_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(config_dict)
    
    @classmethod
    def from_json(cls, json_path):
        """
        Load configuration from JSON file.
        
        Args:
            json_path (str): Path to JSON configuration file
            
        Returns:
            Config: Configuration object
        """
        with open(json_path, 'r') as file:
            config_dict = json.load(file)
        return cls(config_dict)
    
    def save_yaml(self, yaml_path):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path (str): Path to save YAML file
        """
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as file:
            yaml.safe_dump(self.config, file, default_flow_style=False)
    
    def save_json(self, json_path):
        """
        Save configuration to JSON file.
        
        Args:
            json_path (str): Path to save JSON file
        """
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as file:
            json.dump(self.config, file, indent=4)

def load_config(config_path):
    """
    Load configuration from file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Config: Configuration object
    """
    config_path = Path(config_path)
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        return Config.from_yaml(config_path)
    elif config_path.suffix == '.json':
        return Config.from_json(config_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

# Default configurations for each model
DEFAULT_CONFIGS = {
    'model1_vgg16_svm': {
        'model_name': 'VGG16_SVM',
        'feature_extractor': 'vgg16',
        'classifier': 'svm',
        'svm_kernel': 'rbf',
        'svm_C': 1.0,
        'batch_size': 32,
        'input_size': [224, 224, 3],
    },
    'model2_resnet50': {
        'model_name': 'ResNet50',
        'architecture': 'resnet50',
        'num_classes': 2,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 100,
        'optimizer': 'adam',
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy'],
        'input_size': [224, 224, 3],
        'freeze_layers': 100,
    },
    'model3_ensemble': {
        'model_name': 'Ensemble',
        'models': ['alexnet', 'resnet50'],
        'voting_method': 'soft',
        'num_classes': 2,
        'batch_size': 32,
        'epochs': 100,
        'input_size': [224, 224, 3],
    },
    'model4_hybrid_attention': {
        'model_name': 'HybridAttention',
        'backbone': 'resnet50',
        'attention_mechanism': 'cbam',
        'lstm_units': 128,
        'bidirectional': True,
        'num_classes': 2,
        'learning_rate': 0.0001,
        'batch_size': 16,
        'epochs': 150,
        'input_size': [224, 224, 3],
    },
    'model5_parallel_fusion': {
        'model_name': 'ParallelFusion',
        'models': ['inception_resnet_v2', 'xception'],
        'fusion_method': 'concatenate',
        'num_classes': 2,
        'learning_rate': 0.0001,
        'batch_size': 16,
        'epochs': 100,
        'input_size': [299, 299, 3],
    },
    'model6_handcrafted_ann': {
        'model_name': 'HandcraftedANN',
        'features': ['gabor', 'lbp', 'glcm'],
        'ann_layers': [512, 256, 128],
        'activation': 'relu',
        'dropout': 0.5,
        'num_classes': 2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 200,
    }
}