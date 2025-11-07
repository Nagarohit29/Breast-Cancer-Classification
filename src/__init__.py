"""
Breast Cancer Histopathology Classification Project

A comprehensive comparative analysis of six distinct machine learning 
architectures for multi-class classification of breast cancer histopathology images.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import key modules for easy access
from .data.data_loader import DataLoader
from .utils.config import Config
from .utils.helpers import set_random_seed, save_model