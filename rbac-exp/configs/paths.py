"""
Path configurations for RBAC evaluation suite.
"""
import os

# Root paths
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RBAC_EXP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_ROOT = os.path.join(ROOT_PATH, "data")
SELECTED_DATA_PATH = os.path.join(DATA_ROOT, "selected")

# Dataset-specific paths
SPIDER_DATA_PATH = os.path.join(SELECTED_DATA_PATH, "spider")
BIRD_DATA_PATH = os.path.join(SELECTED_DATA_PATH, "bird")
LIVESQLBENCH_DATA_PATH = os.path.join(SELECTED_DATA_PATH, "livesqlbench-full")

# Database paths
SPIDER_DB_PATH = os.path.join(DATA_ROOT, "spider", "database")
BIRD_DB_PATH = os.path.join(DATA_ROOT, "Bird", "dev_20251106", "dev_databases")
LIVESQLBENCH_DB_PATH = os.path.join(DATA_ROOT, "livesqlbench-full-postgresql")

# Output paths
OUTPUT_PATH = os.path.join(RBAC_EXP_PATH, "output")
PRED_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "pred")
EVAL_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "eval_result")
LOG_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "logs")

# Ensure output directories exist
for path in [OUTPUT_PATH, PRED_OUTPUT_PATH, EVAL_OUTPUT_PATH, LOG_OUTPUT_PATH]:
    os.makedirs(path, exist_ok=True)


def get_dataset_paths(dataset: str) -> dict:
    """Get paths for a specific dataset."""
    paths = {
        "spider": {
            "data": SPIDER_DATA_PATH,
            "db": SPIDER_DB_PATH,
        },
        "bird": {
            "data": BIRD_DATA_PATH,
            "db": BIRD_DB_PATH,
        },
        "livesqlbench": {
            "data": LIVESQLBENCH_DATA_PATH,
            "db": LIVESQLBENCH_DB_PATH,
        },
    }
    
    if dataset.lower() not in paths:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(paths.keys())}")
    
    return paths[dataset.lower()]
