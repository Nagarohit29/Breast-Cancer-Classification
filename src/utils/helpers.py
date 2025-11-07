"""
Helper functions and utilities for the breast cancer classification project.
"""

"""
Helper functions and utilities for the breast cancer classification project.
"""

import os
import random
import numpy as np
import importlib
import logging
try:
    import torch
except Exception:
    torch = None
import pickle
import joblib
from datetime import datetime
import logging

def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # If torch is available, use it to set seeds and deterministic flags
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Prefer deterministic algorithms (may slow down training)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        except Exception:
            # Older torch versions may not have use_deterministic_algorithms
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

def setup_logging(log_file=None, level=logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_file (str): Path to log file (optional)
        level (int): Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def save_model(model, filepath, save_format='auto'):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        filepath (str): Path to save the model
        save_format (str): Format to save ('tf', 'h5', 'pkl')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Try formats in order: PyTorch, Keras/TensorFlow (if available), pickle/joblib
    if save_format == 'pytorch' or (hasattr(model, '__module__') and 'torch' in str(type(model)).lower()) or filepath.endswith('.pt') or filepath.endswith('.pth'):
        # PyTorch model
        if torch is None:
            raise RuntimeError('PyTorch is not available to save a torch model')
        torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, filepath)
    elif save_format == 'tf' or filepath.endswith('.h5'):
        # Keras/TensorFlow models (only if keras available)
        try:
            keras = importlib.import_module('tensorflow.keras')
            model.save(filepath)
        except Exception:
            raise RuntimeError('TensorFlow/Keras not available to save tf model')
    elif save_format == 'pkl' or filepath.endswith('.pkl'):
        # For sklearn models
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    elif filepath.endswith('.joblib'):
        # For sklearn models (alternative)
        joblib.dump(model, filepath)
    else:
        raise ValueError(f"Unsupported save format: {save_format}")
    

def load_model(filepath):
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Loaded model object
    """
    # Try PyTorch first
    if (filepath.endswith('.pt') or filepath.endswith('.pth')) and torch is not None:
        # caller must know what to do with the returned state_dict
        try:
            state = torch.load(filepath, map_location='cpu')
            return state
        except Exception:
            pass

    if filepath.endswith('.h5') or (os.path.isdir(filepath) and os.path.exists(os.path.join(filepath, 'saved_model.pb'))):
        # TensorFlow/Keras model
        try:
            keras = importlib.import_module('tensorflow.keras')
            return keras.models.load_model(filepath)
        except Exception:
            raise RuntimeError('TensorFlow/Keras not available to load tf model')
    elif filepath.endswith('.pkl'):
        # Pickle file
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filepath.endswith('.joblib'):
        # Joblib file
        return joblib.load(filepath)
    else:
        raise ValueError(f"Unsupported model format: {filepath}")

def create_experiment_dir(base_dir, experiment_name=None):
    """
    Create a unique directory for an experiment.
    
    Args:
        base_dir (str): Base directory for experiments
        experiment_name (str): Name of the experiment (optional)
        
    Returns:
        str: Path to the created experiment directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir

def count_parameters(model):
    """
    Count the number of parameters in a model.
    
    Args:
        model: Model object
        
    Returns:
        dict: Dictionary with parameter counts
    """
    # Keras-style models
    try:
        if hasattr(model, 'count_params'):
            total_params = model.count_params()
            trainable_params = 0
            if hasattr(model, 'trainable_weights'):
                # Try Keras backend
                try:
                    keras = importlib.import_module('tensorflow.keras')
                    trainable_params = np.sum([keras.backend.count_params(w) for w in model.trainable_weights])
                except Exception:
                    trainable_params = 'N/A'
            non_trainable_params = total_params - trainable_params if isinstance(trainable_params, (int, np.integer)) else 'N/A'
            return {'total': total_params, 'trainable': trainable_params, 'non_trainable': non_trainable_params}
    except Exception:
        pass

    # PyTorch-style models
    if torch is not None and hasattr(model, 'parameters'):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = total - trainable
        return {'total': total, 'trainable': trainable, 'non_trainable': non_trainable}

    return {'total': 'N/A', 'trainable': 'N/A', 'non_trainable': 'N/A'}

def get_class_weights(y_train):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y_train (np.ndarray): Training labels
        
    Returns:
        dict: Class weights dictionary
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    return dict(zip(classes, class_weights))

def format_time(seconds):
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def print_model_summary(model, model_name="Model"):
    """
    Print a summary of model information.
    
    Args:
        model: Model object
        model_name (str): Name of the model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Summary")
    print(f"{'='*50}")
    
    # Parameter count
    param_info = count_parameters(model)
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    print(f"Non-trainable parameters: {param_info['non_trainable']:,}")
    
    # Model architecture (if available)
    if hasattr(model, 'summary'):
        print("\nModel Architecture:")
        model.summary()
    
    print(f"{'='*50}\n")

def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    os.makedirs(directory, exist_ok=True)

def get_gpu_info():
    """
    Get information about available GPUs.
    
    Returns:
        dict: GPU information
    """
    info = {
        'num_gpus': 0,
        'gpu_names': [],
        'memory_growth_enabled': False
    }

    if torch is not None:
        try:
            num = torch.cuda.device_count()
            info['num_gpus'] = num
            for i in range(num):
                try:
                    info['gpu_names'].append(torch.cuda.get_device_name(i))
                except Exception:
                    info['gpu_names'].append(f'GPU:{i}')
            # PyTorch doesn't expose a simple memory growth toggle like TF; leave False
        except Exception:
            pass

    return info