"""
Compatibility shim for data loader API.

This module provides a stable import surface expected by other parts of
the project (DataLoader and preprocess_images). The real implementations
live in other modules (e.g. breakhis_loader.py and preprocessing.py),
so this file imports and re-exports them under the expected names.
"""
from .breakhis_loader import BreaKHisDataLoader
from .preprocessing import run_preprocessing


# Expose DataLoader name expected by other modules
class DataLoader(BreaKHisDataLoader):
    """Alias for BreaKHisDataLoader kept for backwards compatibility."""
    pass


def preprocess_images(*args, **kwargs):
    """Compatibility wrapper for preprocessing.

    Historically other modules call `preprocess_images()`; map that to
    the implemented `run_preprocessing()` function. Any arguments are
    passed through (currently ignored by run_preprocessing).
    """
    return run_preprocessing(*args, **kwargs)
