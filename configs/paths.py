"""
Centralized path configuration for Role-SQL Benchmark.

This module provides a single source of truth for all project paths,
eliminating hardcoded paths and improving reproducibility.
"""

from pathlib import Path
import os


def get_project_root() -> Path:
    """
    Get the project root directory dynamically.
    
    Returns:
        Path: Absolute path to project root
    """
    # Get the directory containing this config file
    current_file = Path(__file__).resolve()
    # Project root is two levels up from configs/paths.py
    project_root = current_file.parent.parent
    return project_root


# Project structure
PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data directories
SPIDER_DIR = DATA_DIR / "spider"
BIRD_DIR = DATA_DIR / "Bird"
LIVESQL_DIR = DATA_DIR / "livesqlbench-base-lite-sqlite"

# Output directories
SPIDER_OUTPUT_DIR = OUTPUTS_DIR / "spider_data"
BIRD_OUTPUT_DIR = OUTPUTS_DIR / "bird_data"
LIVESQL_OUTPUT_DIR = OUTPUTS_DIR / "livesql_data"

# Selected datasets
SELECTED_DIR = DATA_DIR / "selected"
DATASET_INFO_JSON = SELECTED_DIR / "dataset_info.json"

# Create output directories if they don't exist
for output_dir in [OUTPUTS_DIR, SPIDER_OUTPUT_DIR, BIRD_OUTPUT_DIR, LIVESQL_OUTPUT_DIR]:
    output_dir.mkdir(parents=True, exist_ok=True)


def get_database_path(dataset: str, split: str = "dev") -> Path:
    """
    Get database path for a specific dataset.
    
    Args:
        dataset: Dataset name ('spider', 'bird', 'livesql')
        split: Data split ('train', 'dev', 'test')
    
    Returns:
        Path: Database directory path
    """
    dataset = dataset.lower()
    
    if dataset == "spider":
        return SPIDER_DIR / "database"
    elif dataset == "bird":
        if split == "train":
            return BIRD_DIR / "train_20240627" / "train_databases"
        else:
            return BIRD_DIR / "dev_20240627" / "dev_databases"
    elif dataset == "livesql":
        return LIVESQL_DIR
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_output_path(dataset: str, filename: str) -> Path:
    """
    Get output file path for a specific dataset.
    
    Args:
        dataset: Dataset name ('spider', 'bird', 'livesql')
        filename: Output filename
    
    Returns:
        Path: Full output file path
    """
    dataset = dataset.lower()
    
    if dataset == "spider":
        return SPIDER_OUTPUT_DIR / filename
    elif dataset == "bird":
        return BIRD_OUTPUT_DIR / filename
    elif dataset == "livesql":
        return LIVESQL_OUTPUT_DIR / filename
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# Environment variable fallbacks
def get_env_path(var_name: str, default: Path) -> Path:
    """
    Get path from environment variable or use default.
    
    Args:
        var_name: Environment variable name
        default: Default path if env var not set
    
    Returns:
        Path: Resolved path
    """
    env_value = os.getenv(var_name)
    if env_value:
        return Path(env_value)
    return default


if __name__ == "__main__":
    # Test path configuration
    print("=== Role-SQL Benchmark Path Configuration ===")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Outputs Directory: {OUTPUTS_DIR}")
    print(f"\nDataset Paths:")
    print(f"  Spider: {SPIDER_DIR}")
    print(f"  BIRD: {BIRD_DIR}")
    print(f"  LiveSQL: {LIVESQL_DIR}")
    print(f"\nOutput Paths:")
    print(f"  Spider: {SPIDER_OUTPUT_DIR}")
    print(f"  BIRD: {BIRD_OUTPUT_DIR}")
    print(f"  LiveSQL: {LIVESQL_OUTPUT_DIR}")
