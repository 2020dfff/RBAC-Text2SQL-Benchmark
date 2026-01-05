"""
Data loading utilities for RBAC datasets.
Handles loading and parsing of Spider, Bird, and LiveSQLBench datasets.
"""
import json
import os
import glob
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass

from configs.paths import get_dataset_paths
from configs.datasets import get_dataset_config, DatasetConfig
from configs.prompts import format_column_level_prompt, format_crud_level_prompt


@dataclass
class RBACDataItem:
    """Unified data item structure for RBAC evaluation."""
    id: str
    instruction: str           # Schema + policy (without system prompt)
    gold_sql: str              # Ground truth SQL
    is_allowed: bool           # Ground truth permission (True=allowed, False=denied)
    database: str              # Database name
    input: Optional[str] = None        # Question/input text (for prompt building)
    role: Optional[str] = None         # Role name (for structured prompt)
    policy: Optional[Dict[str, Any]] = None  # Structured policy (for structured prompt)
    difficulty: Optional[str] = None   # Difficulty level (Spider/Bird)
    operation: Optional[str] = None    # Operation type (LiveSQLBench)
    category: Optional[str] = None     # Category (LiveSQLBench)
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    
    # Prediction fields (filled during inference)
    prediction: Optional[str] = None
    pred_sql: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "instruction": self.instruction,
            "input": self.input,
            "role": self.role,
            "policy": self.policy,
            "gold_sql": self.gold_sql,
            "is_allowed": self.is_allowed,
            "database": self.database,
            "difficulty": self.difficulty,
            "operation": self.operation,
            "category": self.category,
            "metadata": self.metadata,
            "prediction": self.prediction,
            "pred_sql": self.pred_sql,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RBACDataItem":
        """Create from dictionary."""
        # Handle field name variations between prediction output and dataset
        # predict_local.py outputs: cleaned_sql, raw_response
        # Original format expects: pred_sql, prediction
        pred_sql = data.get("pred_sql") or data.get("cleaned_sql")
        prediction = data.get("prediction") or data.get("raw_response")
        
        # Handle is_allowed vs expected_action
        is_allowed = data.get("is_allowed")
        if is_allowed is None and "expected_action" in data:
            is_allowed = data.get("expected_action", "").upper() == "ALLOW"
        elif is_allowed is None:
            is_allowed = True
        
        return cls(
            id=data.get("id", ""),
            instruction=data.get("instruction", ""),
            input=data.get("input", data.get("question", "")),
            role=data.get("role"),
            policy=data.get("policy"),
            gold_sql=data.get("gold_sql", ""),
            is_allowed=is_allowed,
            database=data.get("database", ""),
            difficulty=data.get("difficulty"),
            operation=data.get("operation"),
            category=data.get("category"),
            metadata=data.get("metadata"),
            prediction=prediction,
            pred_sql=pred_sql,
        )


def _get_nested_field(data: Dict[str, Any], field_path: str) -> Any:
    """Get nested field value using dot notation (e.g., 'metadata.permission')."""
    keys = field_path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def _extract_operation_type(sql: str) -> str:
    """Extract operation type from SQL query."""
    sql_upper = sql.strip().upper()
    
    operations = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]
    for op in operations:
        if sql_upper.startswith(op):
            return op
    
    return "OTHER"


def parse_spider_bird_item(
    raw_item: Dict[str, Any], 
    dataset_config: DatasetConfig, 
    index: int,
    shot_num: int = 0
) -> RBACDataItem:
    """Parse Spider/Bird dataset item to unified format with column-level RBAC prompt."""
    # Get instruction (contains schema + column-level policy)
    # NOTE: Do NOT format prompt here - keep raw instruction
    # Prompt formatting (with system prompt and few-shot) happens in predict stage
    raw_instruction = raw_item.get("instruction", "")
    
    # Get the question
    question = raw_item.get("input", "")
    
    # Get role and policy for structured prompt
    role = raw_item.get("role", "")
    policy = raw_item.get("policy", {})  # Dict: {"table": ["col1", "col2", "*"]}
    
    # Get gold SQL from metadata (the actual SQL to execute)
    gold_sql = _get_nested_field(raw_item, dataset_config.gold_sql_field) or ""
    
    # Get permission (allowed/denied)
    permission_value = _get_nested_field(raw_item, dataset_config.permission_field)
    is_allowed = permission_value == "allowed" if permission_value else True
    
    # Get database name
    database = raw_item.get("db_id", raw_item.get("database", ""))
    
    # Get difficulty
    difficulty = raw_item.get("difficulty", raw_item.get(dataset_config.difficulty_field)) if dataset_config.difficulty_field else None
    
    # Extract operation type from gold SQL
    operation = _extract_operation_type(gold_sql)
    
    return RBACDataItem(
        id=raw_item.get("id", f"{dataset_config.name}_{index}"),
        instruction=raw_instruction,  # Keep raw instruction, not formatted prompt
        input=question,  # Store question separately for prompt building later
        role=role,  # For structured prompt
        policy=policy,  # For structured prompt
        gold_sql=gold_sql,
        is_allowed=is_allowed,
        database=database,
        difficulty=difficulty,
        operation=operation,
        metadata=raw_item.get("metadata"),
    )


def parse_livesqlbench_item(
    raw_item: Dict[str, Any], 
    dataset_config: DatasetConfig, 
    index: int,
    shot_num: int = 0
) -> RBACDataItem:
    """Parse LiveSQLBench dataset item to unified format with CRUD-level RBAC prompt."""
    # Get instruction (contains schema + CRUD policy)
    # NOTE: Do NOT format prompt here - keep raw instruction
    # Prompt formatting (with system prompt and few-shot) happens in predict stage
    raw_instruction = raw_item.get("instruction", "")
    
    # Get the question
    question = raw_item.get("input", "")
    
    # Get role and policy for structured prompt
    role = raw_item.get("role", "")
    policy = raw_item.get("policy", {})  # Dict with CRUD permissions
    
    # Get gold SQL
    gold_sql = raw_item.get("gold_sql", raw_item.get("gold_output", ""))
    
    # Get permission (allowed field is boolean)
    is_allowed = raw_item.get("allowed", True)
    
    # Get database name
    database = raw_item.get("db_id", raw_item.get("database", ""))
    
    # Get category
    category = raw_item.get("category")
    
    # Extract operation type from gold SQL (used for classification)
    operation = _extract_operation_type(gold_sql)
    
    return RBACDataItem(
        id=raw_item.get("id", f"{dataset_config.name}_{index}"),
        instruction=raw_instruction,  # Keep raw instruction
        input=question,  # Store question separately
        role=role,  # For structured prompt
        policy=policy,  # For structured prompt
        gold_sql=gold_sql,
        is_allowed=is_allowed,
        database=database,
        operation=operation,
        category=category,
        metadata={
            "original_item": {k: v for k, v in raw_item.items() 
                            if k not in ["instruction", "gold_sql", "allowed", "db_id"]}
        },
    )


def load_dataset(
    dataset: str,
    data_path: str,
    max_samples: Optional[int] = None,
    shot_num: int = 0,
) -> List[RBACDataItem]:
    """
    Load RBAC dataset and convert to unified format.
    
    Args:
        dataset: Dataset name (spider, bird, livesqlbench)
        data_path: Path to data file (required)
        max_samples: Optional limit on number of samples
        shot_num: Number of few-shot examples (0, 1, 3, or 5)
        
    Returns:
        List of RBACDataItem objects
    """
    config = get_dataset_config(dataset)
    
    # data_path is required
    if not data_path:
        raise ValueError("data_path is required. Please provide the path to the dataset file.")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data file
    print(f"Loading: {data_path}")
    if shot_num > 0:
        print(f"Using {shot_num}-shot prompting")
    
    with open(data_path, "r", encoding="utf-8") as f:
        if data_path.endswith(".jsonl"):
            raw_data = [json.loads(line) for line in f if line.strip()]
        else:
            raw_data = json.load(f)
    
    # Parse each item
    items: List[RBACDataItem] = []
    for idx, raw_item in enumerate(raw_data):
        if max_samples and len(items) >= max_samples:
            break
            
        if dataset.lower() == "livesqlbench":
            item = parse_livesqlbench_item(raw_item, config, len(items), shot_num=shot_num)
        else:
            item = parse_spider_bird_item(raw_item, config, len(items), shot_num=shot_num)
        
        items.append(item)
    
    print(f"Loaded {len(items)} items from {dataset}")
    return items


def load_predictions(prediction_path: str) -> List[RBACDataItem]:
    """
    Load predictions from file.
    
    Args:
        prediction_path: Path to prediction JSON file
        
    Returns:
        List of RBACDataItem objects with predictions
    """
    with open(prediction_path, "r", encoding="utf-8") as f:
        if prediction_path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    
    items = [RBACDataItem.from_dict(item) for item in data]
    print(f"Loaded {len(items)} predictions from {prediction_path}")
    return items


def save_predictions(items: List[RBACDataItem], output_path: str) -> None:
    """
    Save predictions to file.
    
    Args:
        items: List of RBACDataItem objects
        output_path: Path to save predictions
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = [item.to_dict() for item in items]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(items)} predictions to {output_path}")


def iterate_dataset(
    dataset: str,
    data_path: Optional[str] = None,
    batch_size: int = 1,
) -> Iterator[List[RBACDataItem]]:
    """
    Iterate over dataset in batches.
    
    Args:
        dataset: Dataset name
        data_path: Optional custom data path
        batch_size: Batch size
        
    Yields:
        Batches of RBACDataItem objects
    """
    items = load_dataset(dataset, data_path)
    
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
