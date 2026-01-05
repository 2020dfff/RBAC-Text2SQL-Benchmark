"""
Model and Inference Argument Configurations.
Uses dataclasses for clean argument parsing.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class ModelArgs:
    """Arguments for model loading."""
    model_name_or_path: str = field(
        metadata={"help": "Path to local model or HuggingFace model name"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tokenizer, defaults to model_name_or_path"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code for models"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "Torch dtype: auto, float16, bfloat16, float32"}
    )
    device_map: str = field(
        default="auto",
        metadata={"help": "Device mapping: auto, cuda, cpu"}
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether to use low CPU memory mode"}
    )
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Whether to use vLLM for inference"}
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size for vLLM"}
    )


@dataclass
class CloudAPIArgs:
    """Arguments for cloud API inference."""
    api_provider: str = field(
        default="openai",
        metadata={"help": "API provider: openai, anthropic, gemini, deepseek, deepinfra"}
    )
    api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key (can also be set via environment variable)"}
    )
    api_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Custom API base URL"}
    )
    model_name: str = field(
        default="gpt-4",
        metadata={"help": "Model name for API calls"}
    )
    max_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum tokens for response"}
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature"}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling"}
    )
    concurrent_limit: int = field(
        default=10,
        metadata={"help": "Maximum concurrent API requests"}
    )


@dataclass
class DataArgs:
    """Arguments for data loading."""
    dataset: str = field(
        metadata={"help": "Dataset name: spider, bird, livesqlbench"}
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data file (overrides default dataset path)"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to process"}
    )
    split: str = field(
        default="test",
        metadata={"help": "Data split: train, dev, test"}
    )
    
    # Field configuration (auto-detected for standard datasets)
    prompt_field: str = field(
        default="instruction",
        metadata={"help": "Field name for prompt/instruction"}
    )


@dataclass
class InferenceArgs:
    """Arguments for inference."""
    output_path: str = field(
        metadata={"help": "Path to save predictions"}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for inference"}
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum new tokens to generate"}
    )
    do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use sampling"}
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature"}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling"}
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search"}
    )
    save_every: int = field(
        default=100,
        metadata={"help": "Save checkpoint every N samples"}
    )
    resume: bool = field(
        default=True,
        metadata={"help": "Resume from checkpoint if exists"}
    )


@dataclass
class EvaluationArgs:
    """Arguments for evaluation."""
    prediction_path: str = field(
        metadata={"help": "Path to prediction file"}
    )
    dataset: str = field(
        metadata={"help": "Dataset name: spider, bird, livesqlbench"}
    )
    output_path: str = field(
        metadata={"help": "Path to save evaluation results"}
    )
    db_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to database files (overrides default)"}
    )
    
    # Execution-based evaluation
    execute_sql: bool = field(
        default=True,
        metadata={"help": "Whether to execute SQL for verification"}
    )
    timeout: int = field(
        default=30,
        metadata={"help": "SQL execution timeout in seconds"}
    )
    
    # Classification dimension
    classification_by: str = field(
        default="difficulty",
        metadata={"help": "Classification dimension: difficulty or operation"}
    )


# API Provider configurations (aligned with experiments/predict/predict_cloud.py)
API_PROVIDERS = {
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "models": [
            "gpt-5",
            "gpt-5-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ],
        "default_workers": 3,
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-5",
        "models": [
            "claude-sonnet-4-5",
            "claude-sonnet-4",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "default_workers": 5,
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash-exp",
        "models": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro-002",
        ],
        "default_workers": 5,
    },
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        "default_workers": 5,
    },
    "deepinfra": {
        "url": "https://api.deepinfra.com/v1/openai/chat/completions",
        "env_key": "DEEPINFRA_API_KEY",
        "default_model": "google/gemma-3-4b-it",
        "models": [
            "google/gemma-3-4b-it",
            "google/gemma-3-27b-it",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "microsoft/WizardLM-2-8x22B",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        ],
        "default_workers": 10,
    },
}
