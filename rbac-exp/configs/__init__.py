"""
RBAC-Exp Configuration Package

Modules:
- config: Core constants (IGNORE_INDEX, paths, special tokens)
- paths: Dataset and output paths
- datasets: Dataset configuration classes
- prompts: Prompt templates and construction functions
- data_args: Data arguments for training
- model_args: Model and training arguments
- case: Few-shot examples
"""

# Core constants
from configs.config import (
    IGNORE_INDEX,
    ADAPTER_PATH,
    ROOT_PATH,
    RBAC_EXP_PATH,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
)

# Training arguments
from configs.data_args import DataArguments, DatasetAttr, Template
from configs.model_args import (
    ModelArguments,
    FinetuningArguments,
    GeneratingArguments,
    TrainingArguments,
)

# Dataset configuration
from configs.datasets import get_dataset_config, DatasetConfig
from configs.paths import get_dataset_paths

# Prompts
from configs.prompts import (
    COLUMN_LEVEL_SYSTEM_PROMPT,
    CRUD_LEVEL_SYSTEM_PROMPT,
    format_column_level_prompt,
    format_crud_level_prompt,
    format_baseline_prompt,
    build_rbac_prompt,
    get_fewshot_examples,
    get_rbac_type_from_dataset,
)

__all__ = [
    # Constants
    "IGNORE_INDEX",
    "ADAPTER_PATH",
    "ROOT_PATH",
    "RBAC_EXP_PATH",
    # Arguments
    "DataArguments",
    "ModelArguments",
    "FinetuningArguments",
    "GeneratingArguments",
    "TrainingArguments",
    # Dataset
    "get_dataset_config",
    "get_dataset_paths",
    "DatasetConfig",
    # Prompts
    "format_column_level_prompt",
    "format_crud_level_prompt",
    "format_baseline_prompt",
    "build_rbac_prompt",
]
