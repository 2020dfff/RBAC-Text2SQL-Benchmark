"""
Unified random seed management for reproducibility.

This module provides centralized seed setting to ensure
consistent results across different runs and experiments.
"""

import random
import numpy as np
import torch
import os


# Default seed for all experiments
DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED):
    """
    Set random seed for Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_rng(seed: int = None) -> random.Random:
    """
    Get a dedicated random number generator with specific seed.
    
    Args:
        seed: Random seed (default: uses DEFAULT_SEED)
    
    Returns:
        random.Random: Independent random number generator
    """
    if seed is None:
        seed = DEFAULT_SEED
    return random.Random(seed)


def get_numpy_rng(seed: int = None) -> np.random.Generator:
    """
    Get a NumPy random number generator with specific seed.
    
    Args:
        seed: Random seed (default: uses DEFAULT_SEED)
    
    Returns:
        np.random.Generator: Independent NumPy RNG
    """
    if seed is None:
        seed = DEFAULT_SEED
    return np.random.default_rng(seed)


if __name__ == "__main__":
    # Test seed setting
    print("=== Testing Random Seed Configuration ===")
    set_global_seed(42)
    
    print(f"Python random: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
    
    # Test independent RNG
    rng = get_rng(123)
    print(f"Independent RNG: {rng.random()}")
