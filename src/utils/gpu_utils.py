"""
GPU Utilities for PyTorch Models
Handles GPU availability detection and device management
"""

import torch
import logging

def setup_device():
    """
    Setup and return the appropriate device (GPU/CPU) for training
    
    Returns:
        torch.device: The device to use for training
        str: Device information string
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device_info = f"GPU: {gpu_name} ({gpu_memory:.1f}GB)"

        print("CUDA Available - Using GPU")
        print(f"   Device: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f}GB")
        print(f"   CUDA Version: {torch.version.cuda}")

        # Set memory allocation strategy
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        device_info = "CPU (CUDA not available)"
        print("CUDA Not Available - Using CPU")
        print("   This will be significantly slower for training")
    
    return device, device_info

def get_memory_usage():
    """Get current GPU memory usage if available"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
    return "CPU mode - no GPU memory tracking"

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def set_reproducible_training(seed=42):
    """Set random seeds for reproducible training"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Set random seed to {seed} for reproducible training")