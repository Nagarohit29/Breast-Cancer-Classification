"""Small shim to provide a minimal Keras-compatible API surface for projects
that import `src.utils.keras_compat` when TensorFlow isn't available.

This file exposes a few commonly imported symbols (ResNet50, Model, layers)
as lightweight wrappers or stubs so the rest of the codebase can import them
without requiring TensorFlow during development.
"""
from typing import Any

import torchvision.models as _tv_models
import torch.nn as nn


def ResNet50(include_top=True, weights=None, input_shape=None, classes=1000):
    """Return a torchvision ResNet50 model. This is a PyTorch-based replacement
    for Keras' ResNet50 used for import compatibility within the repo.
    """
    model = _tv_models.resnet50(weights=None)
    if not include_top:
        modules = list(model.children())[:-1]
        return nn.Sequential(*modules)
    return model


class Model:
    """Tiny placeholder for a generic model wrapper used by some scripts."""
    def __init__(self, model: Any):
        self.model = model


class layers:
    Dense = nn.Linear
    Conv2D = nn.Conv2d
    Flatten = nn.Flatten
    Dropout = nn.Dropout
    BatchNormalization = nn.BatchNorm2d

__all__ = ['ResNet50', 'Model', 'layers']
