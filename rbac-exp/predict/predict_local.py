#!/usr/bin/env python3
"""
Local model inference for RBAC benchmark evaluation.

Reuses experiments/llm_base/ChatModel for model loading and template handling.
Supports:
- Few-shot prompting (0/2/4/6 shots)
- Multiple templates (chatml, llama2, mistral, etc.)
- Snowflake reasoning mode cleaning
- HuggingFace and vLLM inference

Output format aligned with experiments/predict/predict.py:
- Primary output: .sql file with one cleaned SQL per line
- Secondary output: _detailed.json for debugging/evaluation
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

# Add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_process.data_loader import load_dataset, RBACDataItem
from data_process.response_cleaner import clean_model_response

# Import ChatModel from experiments for model loading and template handling
try:
    from experiments.llm_base.chat_model import ChatModel
    HAS_EXPERIMENTS_CHAT_MODEL = True
except ImportError:
    HAS_EXPERIMENTS_CHAT_MODEL = False

# Import prompt building utilities from rbac-exp configs
from configs.prompts import (
    build_rbac_prompt, 
    get_fewshot_examples,
    get_rbac_type_from_dataset,
    format_baseline_prompt
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments - aligned with experiments/predict/predict.py args."""
    parser = argparse.ArgumentParser(description="Local model inference for RBAC benchmark")
    
    # Model configuration (aligned with experiments)
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to local model or HuggingFace model name")
    parser.add_argument("--template", type=str, default="chatml",
                        choices=["chatml", "llama2", "llama2_zh", "mistral", "gemma", 
                                 "vanilla", "default", "alpaca", "chatglm2", "chatglm3"],
                        help="Chat template for prompt formatting (default: chatml)")
    
    # Input configuration (aligned with experiments)
    parser.add_argument("--predicted_input_filename", type=str, required=True,
                        help="Input JSON file path")
    parser.add_argument("--dataset", type=str, default="spider",
                        choices=["spider", "bird", "livesqlbench"],
                        help="Dataset type for parsing")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process")
    
    # Few-shot configuration
    parser.add_argument("--shot_num", type=int, default=0,
                        choices=[0, 2, 4, 6],
                        help="Number of few-shot examples (0=zero-shot, 2/4/6=few-shot with balanced ALLOW/DENY)")
    
    # Output configuration (aligned with experiments)
    parser.add_argument("--predicted_out_filename", type=str, required=True,
                        help="Output SQL file path (.sql)")
    
    # Generation configuration (aligned with experiments)
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling")
    
    # Snowflake mode for reasoning models
    parser.add_argument("--snowflake_mode", action="store_true",
                        help="Enable Snowflake reasoning mode cleaning (for Arctic-Text2SQL-R1)")
    
    # vLLM configuration (optional)
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for inference (faster but requires vllm package)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    
    # Checkpoint configuration
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N samples")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    
    # Evaluation mode configuration
    parser.add_argument("--mode", type=str, default="rbac",
                        choices=["rbac", "baseline"],
                        help="Evaluation mode: 'rbac' (with RBAC prompts) or 'baseline' (Text2SQL only, no RBAC)")
    
    return parser.parse_args()


class LocalModelInference:
    """
    Local model inference using experiments/llm_base/ChatModel.
    
    Provides consistent interface with experiments framework while adding
    RBAC-specific features like few-shot prompting and snowflake cleaning.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        template: str = "chatml",
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        use_vllm: bool = False,
        tensor_parallel_size: int = 1,
    ):
        """
        Initialize local model using experiments ChatModel or fallback.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            template: Chat template name (chatml, llama2, mistral, etc.)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            use_vllm: Use vLLM for inference
            tensor_parallel_size: GPU parallel size for vLLM
        """
        self.model_name_or_path = model_name_or_path
        self.template = template
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_vllm = use_vllm
        
        if use_vllm:
            self._init_vllm(model_name_or_path, template, tensor_parallel_size)
        elif HAS_EXPERIMENTS_CHAT_MODEL:
            self._init_experiments_chat_model()
        else:
            self._init_fallback_hf(model_name_or_path, template)
    
    def _init_experiments_chat_model(self):
        """Initialize using experiments ChatModel for consistent template handling."""
        import sys
        original_argv = sys.argv
        
        try:
            # Construct minimal args for ChatModel initialization
            sys.argv = [
                "predict_local.py",
                "--model_name_or_path", self.model_name_or_path,
                "--template", self.template,
                "--max_new_tokens", str(self.max_new_tokens),
                "--predicted_input_filename", "dummy.json",  # Required but not used
                "--predicted_out_filename", "dummy.sql",  # Required but not used
            ]
            
            self.chat_model = ChatModel()
            logger.info(f"Loaded model using experiments ChatModel: {self.model_name_or_path}")
            logger.info(f"Template: {self.template}")
        finally:
            sys.argv = original_argv
    
    def _init_vllm(self, model_path: str, template: str, tensor_parallel_size: int):
        """Initialize vLLM engine."""
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
                max_model_len=4096,  # Limit context length to save memory
            )
            self.SamplingParams = SamplingParams
            self.chat_model = None
            logger.info(f"Loaded vLLM model: {model_path} with tensor_parallel_size={tensor_parallel_size}")
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
    
    def _init_fallback_hf(self, model_path: str, template: str):
        """Fallback HuggingFace initialization when experiments not available."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.chat_model = None
        logger.info(f"Loaded fallback HF model: {model_path}")
    
    def generate_single(self, query: str) -> str:
        """
        Generate response for a single query.
        
        Args:
            query: The formatted prompt to send to model
            
        Returns:
            Model response string
        """
        if self.use_vllm:
            return self._generate_vllm_single(query)
        elif self.chat_model is not None:
            # Use experiments ChatModel
            response, (prompt_len, resp_len) = self.chat_model.chat(
                query=query,
                history=[],
            )
            return response
        else:
            # Fallback HF generation
            return self._generate_hf_single(query)
    
    def _generate_vllm_single(self, query: str) -> str:
        """Generate using vLLM."""
        sampling_params = self.SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else 0.01,
            top_p=self.top_p,
        )
        outputs = self.llm.generate([query], sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def generate_batch(self, queries: List[str]) -> List[str]:
        """
        Generate responses for a batch of queries (vLLM only).
        Falls back to sequential generation for non-vLLM modes.
        
        Args:
            queries: List of formatted prompts
            
        Returns:
            List of model response strings
        """
        if self.use_vllm:
            sampling_params = self.SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else 0.01,
                top_p=self.top_p,
            )
            outputs = self.llm.generate(queries, sampling_params)
            return [output.outputs[0].text.strip() for output in outputs]
        else:
            # Fallback to sequential generation
            return [self.generate_single(q) for q in queries]
    
    def _generate_hf_single(self, query: str) -> str:
        """Generate using fallback HuggingFace."""
        import torch
        
        # Apply chat template if available
        messages = [{"role": "user", "content": query}]
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
            )
        except Exception:
            formatted = query
        
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)
        
        prompt_length = inputs.input_ids.shape[1]
        
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        response_ids = outputs[0][prompt_length:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def _select_baseline_item(group_items: List[RBACDataItem]) -> RBACDataItem:
    """
    Select the best item for baseline evaluation from a group of items.
    
    Priority:
    1. SystemManager role (always has full permissions, gold_sql is always valid)
    2. Any item with is_allowed=True
    3. First item (fallback)
    
    This ensures we always get a valid SQL for baseline Text2SQL evaluation.
    """
    # Priority 1: SystemManager role
    for item in group_items:
        if item.role and "systemmanager" in item.role.lower():
            return item
    
    # Priority 2: Any allowed item
    for item in group_items:
        if item.is_allowed:
            return item
    
    # Fallback: first item (shouldn't happen in well-formed datasets)
    return group_items[0]


def prepare_rbac_prompts(
    items: List[RBACDataItem],
    dataset: str,
    shot_num: int = 0,
) -> List[str]:
    """
    Prepare RBAC prompts with optional few-shot examples.
    
    Args:
        items: List of RBAC data items
        dataset: Dataset name (spider, bird, livesqlbench)
        shot_num: Number of few-shot examples (0/2/4/6)
        
    Returns:
        List of formatted prompts
    """
    rbac_type = get_rbac_type_from_dataset(dataset)
    fewshot_examples = get_fewshot_examples(rbac_type, shot_num) if shot_num > 0 else []
    
    prompts = []
    for item in items:
        # Build full prompt with few-shot examples
        prompt = build_rbac_prompt(
            item=item,
            rbac_type=rbac_type,
            fewshot_examples=fewshot_examples,
        )
        prompts.append(prompt)
    
    return prompts


def run_inference(args):
    """Run local model inference on RBAC dataset."""
    import random
    
    mode = getattr(args, 'mode', 'rbac')
    
    logger.info("=" * 60)
    logger.info("RBAC Local Model Inference")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Template: {args.template}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Input: {args.predicted_input_filename}")
    logger.info(f"Output: {args.predicted_out_filename}")
    logger.info(f"Few-shot: {args.shot_num}")
    logger.info(f"Snowflake mode: {args.snowflake_mode}")
    logger.info("=" * 60)
    
    # Initialize model
    model = LocalModelInference(
        model_name_or_path=args.model_name_or_path,
        template=args.template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    # Load dataset
    items = load_dataset(args.dataset, args.predicted_input_filename, args.max_samples)
    logger.info(f"Loaded {len(items)} samples from {args.predicted_input_filename}")
    
    # Baseline mode: deduplicate by instance_id (livesqlbench) or input (spider/bird)
    # IMPORTANT: Select SystemManager role entries (always allowed) instead of random selection
    if mode == "baseline":
        logger.info("Baseline mode: Deduplicating samples by selecting SystemManager role...")
        
        if args.dataset.lower() == "livesqlbench":
            # LiveSQLBench: group by instance_id
            instance_groups = {}
            for item in items:
                key = item.instance_id or item.id
                if key not in instance_groups:
                    instance_groups[key] = []
                instance_groups[key].append(item)
            
            deduped_items = [_select_baseline_item(group) for group in instance_groups.values()]
            logger.info(f"Deduplicated: {len(items)} -> {len(deduped_items)} unique instance_ids")
            items = deduped_items
        else:
            # Spider/Bird: group by input (question)
            question_groups = {}
            for item in items:
                key = item.input or item.id
                if key not in question_groups:
                    question_groups[key] = []
                question_groups[key].append(item)
            
            deduped_items = [_select_baseline_item(group) for group in question_groups.values()]
            logger.info(f"Deduplicated: {len(items)} -> {len(deduped_items)} unique questions")
            items = deduped_items
    
    # Prepare prompts based on mode
    if mode == "baseline":
        logger.info("Using BASELINE prompts (Text2SQL only, no RBAC)")
        prompts = [format_baseline_prompt(item, args.dataset, args.shot_num) for item in items]
    else:
        logger.info("Using RBAC prompts (with role/policy)")
        prompts = prepare_rbac_prompts(items, args.dataset, args.shot_num)
    
    # Setup output paths
    sql_output_path = args.predicted_out_filename
    detailed_output_path = sql_output_path.replace(".sql", "_detailed.json")
    os.makedirs(os.path.dirname(sql_output_path) or ".", exist_ok=True)
    
    # Handle resumption
    start_idx = 0
    results = []
    if args.resume and os.path.exists(detailed_output_path):
        try:
            with open(detailed_output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            start_idx = len(results)
            logger.info(f"Resuming from sample {start_idx}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    
    # Determine cleaner mode
    cleaner_mode = "snowflake" if args.snowflake_mode else "default"
    
    # Run inference
    iterator = list(enumerate(zip(prompts, items)))
    if tqdm:
        iterator = tqdm(iterator, desc="Inference", initial=start_idx)
    
    for idx, (prompt, item) in iterator:
        if idx < start_idx:
            continue
        
        try:
            # Generate response
            response = model.generate_single(prompt)
            
            # Clean response (extract SQL or detect refusal)
            cleaned_sql = clean_model_response(response, mode=cleaner_mode)
            
            # Get question from input field or metadata
            question_text = getattr(item, 'input', '') or ''
            if not question_text and item.metadata:
                question_text = item.metadata.get('question', '') or item.metadata.get('input', '')
            
            # Get expected action (ALLOW/DENY)
            expected_action = "ALLOW" if item.is_allowed else "DENY"
            
            # Store result
            result = {
                "id": item.id,
                "question": question_text[:500] if question_text else "",
                "gold_sql": item.gold_sql,
                "expected_action": expected_action,
                "raw_response": response,
                "cleaned_sql": cleaned_sql,
                "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            }
            results.append(result)
            
            # Save checkpoint periodically
            if (idx + 1) % args.save_every == 0:
                _save_checkpoint(results, sql_output_path, detailed_output_path)
                logger.info(f"Checkpoint saved at sample {idx + 1}")
                
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Get question from input field or metadata
            question_text = getattr(item, 'input', '') or ''
            if not question_text and item.metadata:
                question_text = item.metadata.get('question', '') or item.metadata.get('input', '')
            # Get expected action (ALLOW/DENY)
            expected_action = "ALLOW" if item.is_allowed else "DENY"
            result = {
                "id": item.id,
                "question": question_text[:500] if question_text else "",
                "gold_sql": item.gold_sql,
                "expected_action": expected_action,
                "raw_response": f"ERROR: {str(e)}",
                "cleaned_sql": "Sorry, I cannot answer.",
                "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            }
            results.append(result)
    
    # Final save
    _save_checkpoint(results, sql_output_path, detailed_output_path)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Inference Complete!")
    logger.info(f"Total samples: {len(results)}")
    logger.info(f"SQL output: {sql_output_path}")
    logger.info(f"Detailed output: {detailed_output_path}")
    logger.info("=" * 60)
    
    return sql_output_path, detailed_output_path


def _save_checkpoint(results: List[Dict], sql_path: str, detailed_path: str):
    """Save results to both SQL file and detailed JSON."""
    # Save SQL file (one cleaned SQL per line, aligned with experiments)
    with open(sql_path, "w", encoding="utf-8") as f:
        for result in results:
            sql = result["cleaned_sql"]
            # Replace newlines with spaces (aligned with experiments/predict/predict.py)
            f.write(sql.replace("\n", " ") + "\n")
    
    # Save detailed JSON for debugging/evaluation
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
