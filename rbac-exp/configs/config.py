"""
Core configuration constants for RBAC-Exp training and evaluation.
Cleaned version - only essential constants are kept.
"""
import os

# =============================================================================
# Path Configuration
# =============================================================================
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RBAC_EXP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output paths for RBAC-Exp
OUTPUT_PATH = os.path.join(RBAC_EXP_PATH, "output")
ADAPTER_PATH = os.path.join(OUTPUT_PATH, "adapter")
MERGED_MODELS_PATH = os.path.join(OUTPUT_PATH, "merged_models")

# Ensure output directories exist
os.makedirs(ADAPTER_PATH, exist_ok=True)
os.makedirs(MERGED_MODELS_PATH, exist_ok=True)

# =============================================================================
# Model Training Constants
# =============================================================================
IGNORE_INDEX = -100  # Used for label masking in training

# Special tokens
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# Layer normalization names (for prepare_model_for_training)
LAYERNORM_NAMES = ["norm", "ln_f", "ln_attn", "ln_mlp"]

# File type mapping
EXT2TYPE = {"csv": "csv", "json": "json", "jsonl": "json", "txt": "text"}

# =============================================================================
# Logging and Checkpoint Constants
# =============================================================================
LOG_FILE_NAME = "trainer_log.jsonl"
VALUE_HEAD_FILE_NAME = "value_head.bin"
FINETUNING_ARGS_NAME = "finetuning_args.json"
