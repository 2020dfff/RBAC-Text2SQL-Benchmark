"""
Dataset configurations for RBAC evaluation.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    name: str
    permission_field: str  # Field name for ground truth
    gold_sql_field: str    # Field name for gold SQL
    difficulty_field: Optional[str] = None
    category_field: Optional[str] = None
    db_type: str = "sqlite"  # sqlite or postgresql
    
    # Classification dimensions for evaluation
    classification_dimension: str = "difficulty"  # difficulty or operation
    classification_levels: List[str] = field(default_factory=list)


# Dataset configurations
SPIDER_CONFIG = DatasetConfig(
    name="spider",
    permission_field="metadata.permission",  # "allowed" or "denied"
    gold_sql_field="metadata.gold_sql",
    difficulty_field="difficulty",
    db_type="sqlite",
    classification_dimension="difficulty",
    classification_levels=["easy", "medium", "hard", "extra"],
)

BIRD_CONFIG = DatasetConfig(
    name="bird",
    permission_field="metadata.permission",  # "allowed" or "denied"
    gold_sql_field="metadata.gold_sql",
    difficulty_field="difficulty",
    db_type="sqlite",
    classification_dimension="difficulty",
    classification_levels=["simple", "moderate", "challenging"],
)

LIVESQLBENCH_CONFIG = DatasetConfig(
    name="livesqlbench",
    permission_field="allowed",  # True or False
    gold_sql_field="gold_sql",
    category_field="category",
    db_type="postgresql",
    classification_dimension="operation",
    classification_levels=["SELECT", "CREATE", "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "OTHER"],
)


def get_dataset_config(dataset: str) -> DatasetConfig:
    """Get configuration for a specific dataset."""
    configs = {
        "spider": SPIDER_CONFIG,
        "bird": BIRD_CONFIG,
        "livesqlbench": LIVESQLBENCH_CONFIG,
    }
    
    if dataset.lower() not in configs:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(configs.keys())}")
    
    return configs[dataset.lower()]
