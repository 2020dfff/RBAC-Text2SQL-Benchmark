"""
Data utilities for Column-Level RBAC SFT training.

Key differences from experiments/data_process/data_utils.py:
1. Uses Column-Level RBAC prompts (schema + column-level policy in instruction field)
2. No separate role/tables fields - instruction is self-contained
3. Simpler data loading - instruction already has everything needed

Training data format:
{
    "db_id": "book_2",
    "instruction": "##Instruction:\nDatabase: book_2\n...schema...\n##Role Access Policy (Column-Level):\nRole: CatalogManager\nAccessible Columns: ...",
    "input": "How many books are there?",
    "output": "SELECT count(*) FROM book",  # or "Sorry, I cannot answer." for denied
    "role": "CatalogManager",
    "policy": {...},
    "difficulty": "easy",
    "metadata": {...}
}
"""
import os
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# Add parent directories to path
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
RBAC_EXP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RBAC_EXP_PATH)

# Import from local configs (copied from experiments)
from configs.config import IGNORE_INDEX
from llm_base.loggings import get_logger

# Import Column-Level RBAC prompts
from configs.prompts import COLUMN_LEVEL_RBAC_PROMPT_DICT, COLUMN_LEVEL_SYSTEM_PROMPT

logger = get_logger(__name__)


# =============================================================================
# Dataset Info Management
# =============================================================================

def get_dataset_info_path() -> str:
    """Get the path to dataset_info.json for rbac-exp."""
    return os.path.join(RBAC_EXP_PATH, "configs", "dataset_info.json")


def load_dataset_info() -> Dict[str, Any]:
    """Load dataset configuration from dataset_info.json."""
    info_path = get_dataset_info_path()
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    
    with open(info_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_dataset_path(dataset_name: str) -> str:
    """Resolve dataset file path from dataset name."""
    dataset_info = load_dataset_info()
    
    if dataset_name not in dataset_info:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_info.keys())}")
    
    file_name = dataset_info[dataset_name]["file_name"]
    
    # Resolve relative path from rbac-exp/configs/
    config_dir = os.path.dirname(get_dataset_info_path())
    full_path = os.path.normpath(os.path.join(config_dir, file_name))
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Dataset file not found: {full_path}")
    
    return full_path


# =============================================================================
# Prompt Extraction Functions
# =============================================================================

def extract_column_level_prompt(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract and format Column-Level RBAC prompt from training data.
    
    The instruction field already contains schema + column-level policy,
    so we just format it with the system prompt prefix.
    
    Args:
        example: Dict with 'instruction', 'input', 'output' fields
        
    Returns:
        Dict with 'prompt' key containing formatted prompt
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    
    if input_text:
        prompt_template = COLUMN_LEVEL_RBAC_PROMPT_DICT["prompt_input"]
        prompt = prompt_template.format(instruction=instruction, input=input_text)
    else:
        prompt_template = COLUMN_LEVEL_RBAC_PROMPT_DICT["prompt_no_input"]
        prompt = prompt_template.format(instruction=instruction)
    
    return {"prompt": prompt}


# =============================================================================
# Dataset Loading
# =============================================================================

def load_sft_dataset(
    dataset_name: str,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load and prepare Column-Level RBAC dataset for SFT training.
    
    Args:
        dataset_name: Name of dataset in dataset_info.json
        max_samples: Optional limit on number of samples
        
    Returns:
        HuggingFace Dataset with 'prompt', 'query', 'response' columns
    """
    # Resolve path and load
    dataset_path = resolve_dataset_path(dataset_name)
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Load JSON dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if max_samples and max_samples < len(raw_data):
        raw_data = raw_data[:max_samples]
        logger.info(f"Limiting to {max_samples} samples")
    
    logger.info(f"Loaded {len(raw_data)} samples")
    
    # Get column mapping from dataset_info
    dataset_info = load_dataset_info()
    columns = dataset_info[dataset_name].get("columns", {})
    
    prompt_col = columns.get("prompt", "instruction")
    query_col = columns.get("query", "input")
    response_col = columns.get("response", "output")
    
    # Transform data for training
    processed_data = []
    for item in raw_data:
        # Extract prompt using Column-Level template
        prompt_result = extract_column_level_prompt(item)
        
        processed_data.append({
            "prompt": prompt_result["prompt"],
            "query": item.get(query_col, ""),
            "response": item.get(response_col, ""),
            # Keep metadata for debugging
            "db_id": item.get("db_id", ""),
            "role": item.get("role", ""),
            "difficulty": item.get("difficulty", ""),
        })
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(processed_data)
    
    # Log sample for verification
    if len(dataset) > 0:
        logger.info("=" * 80)
        logger.info("Sample from loaded dataset:")
        logger.info(f"Prompt (first 500 chars): {dataset[0]['prompt'][:500]}...")
        logger.info(f"Query: {dataset[0]['query']}")
        logger.info(f"Response: {dataset[0]['response']}")
        logger.info("=" * 80)
    
    return dataset


# =============================================================================
# Template Management (aligned with experiments/data_process/data_utils.py)
# =============================================================================

# Import Template classes from configs/data_args.py (contains encode_oneturn, etc.)
from configs.data_args import Template, Llama2Template

# Template registry
templates: Dict[str, Template] = {}


def get_template_and_fix_tokenizer(
    name: str, tokenizer: "PreTrainedTokenizer"
) -> Template:
    """Get template and fix tokenizer special tokens."""
    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)
    
    additional_special_tokens = template.stop_words
    
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))
    
    if name is None:
        return None
    
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=additional_special_tokens),
        replace_additional_special_tokens=False,
    )
    return template


def register_template(
    name: str,
    prefix: List[Union[str, Dict[str, str]]],
    prompt: List[Union[str, Dict[str, str]]],
    system: str,
    sep: List[Union[str, Dict[str, str]]],
    stop_words: Optional[List[str]] = [],
    use_history: Optional[bool] = True,
) -> None:
    """Register a new template."""
    template_class = Llama2Template if "llama2" in name else Template
    templates[name] = template_class(
        prefix=prefix,
        prompt=prompt,
        system=system,
        sep=sep,
        stop_words=stop_words,
        use_history=use_history,
    )


# =============================================================================
# Register Templates (copied from experiments/data_process/data_utils.py)
# =============================================================================

r"""
Supports language model inference without histories.
"""
register_template(
    name="vanilla",
    prefix=[],
    prompt=["{{query}}"],
    system="",
    sep=[],
    use_history=False,
)

r"""
Supports language model for mistral sqlcoder-7b
"""
register_template(
    name="mistral",
    prefix=["{{system}}"],
    prompt=["[INST] {{query}} [/INST]"],
    system="",
    sep=[],
)

r"""
Default template.
"""
register_template(
    name="default",
    prefix=["{{system}}"],
    prompt=["Human: {{query}}\nAssistant: "],
    system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    sep=["\n"],
)

r"""
Supports: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
"""
register_template(
    name="llama2",
    prefix=["<<SYS>>\n{{system}}\n<</SYS>>\n\n"],
    prompt=["[INST] {{query}} [/INST] "],
    system=(
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe.  "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    ),
    sep=[],
)

r"""
Supports: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
"""
register_template(
    name="llama2_zh",
    prefix=["<<SYS>>\n{{system}}\n<</SYS>>\n\n"],
    prompt=["[INST] {{query}} [/INST] "],
    system="You are a helpful assistant. 你是一个乐于助人的助手。",
    sep=[],
)

r"""
Supports: https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
"""
register_template(
    name="alpaca",
    prefix=["{{system}}"],
    prompt=["### Instruction:\n{{query}}\n\n### Response:\n"],
    system=(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    ),
    sep=["\n\n"],
)

r"""
Supports: https://huggingface.co/Qwen/Qwen-7B-Chat
"""
register_template(
    name="chatml",
    prefix=[{"token": "<|im_start|>"}, "system\n{{system}}", {"token": "<|im_end|>"}],
    prompt=[
        {"token": "<|im_start|>"},
        "user\n{{query}}",
        {"token": "<|im_end|>"},
        "\n",
        {"token": "<|im_start|>"},
        "assistant\n",
    ],
    system="You are a helpful assistant.",
    sep=["\n"],
    stop_words=["<|im_end|>"],
)

r"""
Supports: https://huggingface.co/THUDM/chatglm2-6b
"""
register_template(
    name="chatglm2",
    prefix=[{"token": "[gMASK]"}, {"token": "sop"}, "{{system}}"],
    prompt=["[Round {{idx}}]\n\n问：{{query}}\n\n答："],
    system="",
    sep=["\n\n"],
)

r"""
Supports: https://huggingface.co/THUDM/chatglm3-6b
"""
register_template(
    name="chatglm3",
    prefix=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        {"token": "<|system|>"},
        "\n",
        "{{system}}",
    ],
    prompt=[
        {"token": "<|user|>"},
        "\n",
        "{{query}}",
        {"token": "<|assistant|>"},
        "\n",
    ],
    system=(
        "You are ChatGLM3, a large language model trained by Zhipu.AI. "
        "Follow the user's instructions carefully. Respond using markdown."
    ),
    sep=[],
    stop_words=["<|user|>", "<|observation|>"],
)

r"""
Supports: https://huggingface.co/google/gemma-2-9b-it
"""
register_template(
    name="gemma",
    prefix=[],
    prompt=[
        {"token": "<start_of_turn>"},
        "user\n{{query}}",
        {"token": "<end_of_turn>"},
        "\n",
        {"token": "<start_of_turn>"},
        "model\n",
    ],
    system="",
    sep=["\n"],
    stop_words=["<end_of_turn>", "<eos>"],
)


# =============================================================================
# Preprocessing Functions
# =============================================================================

def preprocess_supervised_dataset(
    dataset: Dataset,
    tokenizer: "PreTrainedTokenizer",
    template: Template,
    max_source_length: int = 2048,
    max_target_length: int = 512,
) -> Dataset:
    """
    Preprocess dataset for supervised fine-tuning.
    
    Tokenizes prompts and responses, creating input_ids and labels.
    Labels for the prompt portion are set to IGNORE_INDEX.
    
    Args:
        dataset: Dataset with 'prompt', 'query', 'response' columns
        tokenizer: HuggingFace tokenizer
        template: Chat template for formatting
        max_source_length: Max length for source (prompt + query)
        max_target_length: Max length for target (response)
        
    Returns:
        Dataset with 'input_ids', 'attention_mask', 'labels' columns
    """
    
    def tokenize_function(examples):
        """Tokenize a batch of examples using template.encode_oneturn()."""
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for i in range(len(examples["prompt"])):
            prompt = examples["prompt"][i]
            query = examples["query"][i]
            response = examples["response"][i]
            
            # For Column-Level RBAC, the 'prompt' field contains the full instruction with schema
            # The 'query' field contains the user's question
            # We combine them as the input to the model
            full_query = prompt + "\n" + query
            
            # Use template's encode_oneturn to properly format and tokenize
            # This handles all the template-specific formatting (prefix, prompt, sep, etc.)
            source_ids, target_ids = template.encode_oneturn(
                tokenizer, 
                query=full_query,
                resp=response,
                history=None,
                system=None  # Use template's default system prompt
            )
            
            # Truncate if necessary
            if len(source_ids) > max_source_length:
                source_ids = source_ids[:max_source_length]
            if len(target_ids) > max_target_length:
                target_ids = target_ids[:max_target_length]
            
            # Concatenate source and target for causal LM training
            input_ids = source_ids + target_ids
            attention_mask = [1] * len(input_ids)
            
            # Create labels: IGNORE_INDEX for source, actual tokens for target
            labels = [IGNORE_INDEX] * len(source_ids) + target_ids
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
        
        return model_inputs
    
    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    logger.info(f"Tokenized {len(tokenized_dataset)} samples")
    
    # Log sample
    if len(tokenized_dataset) > 0:
        sample = tokenized_dataset[0]
        logger.info("=" * 80)
        logger.info("Sample tokenized data:")
        logger.info(f"Input length: {len(sample['input_ids'])} tokens")
        logger.info(f"Decoded input: {tokenizer.decode(sample['input_ids'][:200])}...")
        
        # Decode labels (replace IGNORE_INDEX with pad token for decoding)
        label_tokens = [t if t != IGNORE_INDEX else tokenizer.pad_token_id for t in sample['labels']]
        logger.info(f"Decoded labels: {tokenizer.decode(label_tokens)}")
        logger.info("=" * 80)
    
    return tokenized_dataset


def split_dataset(
    dataset: Dataset,
    eval_ratio: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """Split dataset into train and eval sets."""
    if eval_ratio > 0:
        split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
        return {
            "train_dataset": split["train"],
            "eval_dataset": split["test"],
        }
    else:
        return {
            "train_dataset": dataset,
            "eval_dataset": None,
        }


# =============================================================================
# High-Level API
# =============================================================================

def get_dataset(
    dataset_name: str,
    tokenizer: "PreTrainedTokenizer",
    template_name: str = "chatml",
    max_source_length: int = 2048,
    max_target_length: int = 512,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load and preprocess dataset for SFT training.
    
    Args:
        dataset_name: Name from dataset_info.json
        tokenizer: HuggingFace tokenizer
        template_name: Template name (chatml, llama2, gemma, etc.)
        max_source_length: Max source length
        max_target_length: Max target length
        max_samples: Optional sample limit
        
    Returns:
        Tokenized Dataset ready for training
    """
    # Load raw dataset
    dataset = load_sft_dataset(dataset_name, max_samples)
    
    # Get template
    template = get_template_and_fix_tokenizer(template_name, tokenizer)
    
    # Preprocess
    tokenized_dataset = preprocess_supervised_dataset(
        dataset,
        tokenizer,
        template,
        max_source_length,
        max_target_length,
    )
    
    return tokenized_dataset


if __name__ == "__main__":
    # Test loading
    print("Testing data loading...")
    
    # Test dataset info
    info = load_dataset_info()
    print(f"Available datasets: {list(info.keys())}")
    
    # Test path resolution
    for name in info.keys():
        try:
            path = resolve_dataset_path(name)
            print(f"  {name} -> {path}")
        except FileNotFoundError as e:
            print(f"  {name} -> NOT FOUND: {e}")
    
    # Test loading a dataset (without tokenizer)
    try:
        dataset = load_sft_dataset("column_level_rbac_spider_train", max_samples=5)
        print(f"\nLoaded dataset with {len(dataset)} samples")
        print(f"Columns: {dataset.column_names}")
        print(f"\nSample 0:")
        print(f"  Prompt: {dataset[0]['prompt'][:300]}...")
        print(f"  Response: {dataset[0]['response']}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
