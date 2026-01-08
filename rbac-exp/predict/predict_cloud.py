#!/usr/bin/env python3
"""
Cloud API inference for RBAC benchmark evaluation.
Supports OpenAI, Anthropic, Google Gemini, DeepSeek, and DeepInfra.

Output format aligned with experiments/predict/predict_cloud.py:
- Primary output: .sql file with one cleaned SQL per line
- Secondary output: _detailed.json for debugging/evaluation
"""
import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(ROOT_PATH, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system environment variables

from data_process.data_loader import load_dataset, RBACDataItem
from predict.cloud_client import CloudAPIClient, PROVIDERS
from configs.prompts import (
    format_column_level_prompt, 
    format_crud_level_prompt, 
    get_rbac_type_from_dataset,
    build_structured_prompt,
    format_baseline_prompt
)

# Import response cleaner from experiments (for consistent SQL extraction)
try:
    from experiments.predict.response_cleaner import clean_model_response
except ImportError:
    # Fallback to local cleaner if experiments not available
    from data_process.response_cleaner import clean_sql_response as clean_model_response

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments - aligned with experiments/scripts/predict_cloud.sh"""
    parser = argparse.ArgumentParser(description="Cloud API inference for RBAC benchmark")
    
    # API configuration
    parser.add_argument("--provider", type=str, required=True,
                        choices=list(PROVIDERS.keys()),
                        help="API provider (openai, anthropic, deepseek, deepinfra, gemini)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: provider's default)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (can also use environment variable)")
    
    # Input configuration (aligned with experiments)
    parser.add_argument("--input", type=str, required=True,
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
    
    # Prompt mode configuration
    parser.add_argument("--structured", action="store_true",
                        help="Use structured prompt format (extracts schema from instruction, uses JSON policy)")
    
    # Evaluation mode configuration
    parser.add_argument("--mode", type=str, default="rbac",
                        choices=["rbac", "baseline"],
                        help="Evaluation mode: 'rbac' (with RBAC prompts) or 'baseline' (Text2SQL only, no RBAC)")
    
    # Output configuration (aligned with experiments)
    parser.add_argument("--output", type=str, required=True,
                        help="Output SQL file path")
    
    # Inference configuration (aligned with experiments)
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum tokens for response")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of concurrent workers (default: provider default)")
    parser.add_argument("--rate_limit", type=float, default=1.0,
                        help="Rate limit delay between requests (seconds)")
    
    return parser.parse_args()


class ProgressTracker:
    """Thread-safe progress tracking with progress bar."""
    
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.errors = 0
        self.start_time = time.time()
        
        if tqdm is not None:
            self.pbar = tqdm(total=total, desc="Processing", unit="sample")
        else:
            self.pbar = None
            logger.info(f"Processing {total} samples...")
    
    def update(self, success: bool = True):
        self.completed += 1
        if not success:
            self.errors += 1
        if self.pbar:
            self.pbar.update(1)
    
    def close(self):
        if self.pbar:
            self.pbar.close()


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


def run_inference(args) -> Dict[str, Any]:
    """
    Run cloud inference aligned with experiments workflow.
    
    Supports two modes:
    - 'rbac': Full RBAC evaluation with role/policy prompts
    - 'baseline': Text2SQL only (no RBAC), deduplicated by instance_id/input
    
    Output: 
    - Primary: .sql file with one cleaned SQL per line
    - Secondary: _detailed.json for debugging/evaluation
    """
    import random
    
    mode = getattr(args, 'mode', 'rbac')
    logger.info(f"Starting cloud inference with provider: {args.provider}, mode: {mode}")
    
    # Initialize client
    client = CloudAPIClient(
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
    )
    
    # Determine worker count
    workers = args.workers
    if workers is None:
        workers = PROVIDERS[args.provider].default_workers
        logger.info(f"Using default {workers} concurrent workers for {args.provider}")
    else:
        logger.info(f"Using {workers} concurrent workers (user specified)")
    
    # Load dataset with few-shot support
    shot_num = getattr(args, 'shot_num', 0)
    if shot_num > 0:
        logger.info(f"Using {shot_num}-shot prompting")
    items = load_dataset(args.dataset, args.input, args.max_samples, shot_num=shot_num)
    
    if not items:
        logger.error("No data loaded!")
        return {"error": "No data loaded"}
    
    # Baseline mode: deduplicate by instance_id (livesqlbench) or input (spider/bird)
    # IMPORTANT: Select SystemManager role entries (always allowed) instead of random selection
    if mode == "baseline":
        logger.info("Baseline mode: Deduplicating samples by selecting SystemManager role...")
        
        if args.dataset.lower() == "livesqlbench":
            # LiveSQLBench: group by instance_id, prefer SystemManager role
            instance_groups = {}
            for item in items:
                key = item.instance_id or item.id
                if key not in instance_groups:
                    instance_groups[key] = []
                instance_groups[key].append(item)
            
            # Select SystemManager if available, otherwise first allowed item
            deduped_items = []
            for instance_id, group_items in instance_groups.items():
                selected = _select_baseline_item(group_items)
                deduped_items.append(selected)
            
            logger.info(f"Deduplicated: {len(items)} -> {len(deduped_items)} unique instance_ids")
            items = deduped_items
        else:
            # Spider/Bird: group by input (question), prefer SystemManager role
            question_groups = {}
            for item in items:
                key = item.input or item.id
                if key not in question_groups:
                    question_groups[key] = []
                question_groups[key].append(item)
            
            # Select SystemManager if available, otherwise first allowed item
            deduped_items = []
            for question, group_items in question_groups.items():
                selected = _select_baseline_item(group_items)
                deduped_items.append(selected)
            
            logger.info(f"Deduplicated: {len(items)} -> {len(deduped_items)} unique questions")
            items = deduped_items
    
    # Determine RBAC type for prompt formatting
    rbac_type = get_rbac_type_from_dataset(args.dataset)
    use_structured = getattr(args, 'structured', False)
    
    if mode == "baseline":
        logger.info(f"Using BASELINE prompt format (Text2SQL only, no RBAC)")
    elif use_structured:
        logger.info(f"Using STRUCTURED prompt format for dataset: {args.dataset}")
    else:
        logger.info(f"Using ORIGINAL prompt format ({rbac_type}) for dataset: {args.dataset}")
    
    # Build prompts for each item
    prompts = []
    for item in items:
        if mode == "baseline":
            # Baseline mode: no RBAC information
            prompt = format_baseline_prompt(item, args.dataset, shot_num)
        elif use_structured:
            # Structured mode: extract schema, use JSON policy
            prompt = build_structured_prompt(item, args.dataset, shot_num)
        elif rbac_type == "crud_level":
            prompt = format_crud_level_prompt(item.instruction, item.input or "", shot_num)
        else:
            prompt = format_column_level_prompt(item.instruction, item.input or "", shot_num)
        prompts.append(prompt)
    
    # Setup output path
    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {output_path}")
    
    # Initialize progress tracking
    progress = ProgressTracker(len(items))
    results_dict: Dict[int, str] = {}  # index -> raw response
    start_time = time.time()
    
    try:
        # Run concurrent predictions
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, prompt in enumerate(prompts):
                future = executor.submit(
                    client.call,
                    prompt,
                    rate_limit_delay=args.rate_limit,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                futures[future] = idx
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    response = future.result()
                    results_dict[idx] = response
                    progress.update(success=True)
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    results_dict[idx] = f"Error: {str(e)}"
                    progress.update(success=False)
                    
    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, stopping...")
        return {"error": "Interrupted by user"}
    finally:
        progress.close()
    
    # Reorder results by index
    raw_results = []
    for i in range(len(items)):
        if i in results_dict:
            raw_results.append(results_dict[i])
        else:
            raw_results.append("Error: No result")
    
    # Clean results and save as .sql file (one cleaned SQL per line)
    cleaned_results = []
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, result in enumerate(raw_results):
                # Clean and format output using experiments' cleaner
                cleaned_result = clean_model_response(result)
                if not cleaned_result:
                    cleaned_result = "Empty response"
                cleaned_results.append(cleaned_result)
                f.write(cleaned_result + "\n")
        logger.info(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        raise
    
    # Also save detailed JSON for debugging/evaluation
    json_output_path = output_path.replace(".sql", "_detailed.json")
    try:
        detailed_data = []
        for item, raw_response, cleaned_sql in zip(items, raw_results, cleaned_results):
            detailed_data.append({
                "id": item.id,
                "instance_id": item.instance_id,  # For baseline mode tracking
                "database": item.database,
                "gold_sql": item.gold_sql,
                "is_allowed": item.is_allowed,
                "pred_sql": cleaned_sql,
                "raw_response": raw_response,
                "difficulty": item.difficulty,
                "mode": mode,  # Track evaluation mode
                "metadata": item.metadata,
            })
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to: {json_output_path}")
    except Exception as e:
        logger.warning(f"Failed to save detailed JSON: {e}")
    
    # Calculate statistics
    total_time = time.time() - start_time
    error_count = sum(1 for r in cleaned_results if r.startswith("Error:"))
    sorry_count = sum(1 for r in cleaned_results if "Sorry" in r and not r.startswith("Error:"))
    success_count = len(cleaned_results) - error_count - sorry_count
    
    stats = {
        "provider": args.provider,
        "model": client.model_name,
        "mode": mode,
        "total_samples": len(items),
        "successful_samples": success_count,
        "sorry_responses": sorry_count,
        "system_errors": error_count,
        "workers": workers,
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_sample": round(total_time / len(items), 2),
        "throughput_samples_per_second": round(len(items) / total_time, 2),
        "output_file": output_path,
    }
    
    logger.info(f"Cloud inference completed successfully")
    logger.info(f"Statistics: {stats}")
    
    return stats


def main():
    args = parse_args()
    
    try:
        stats = run_inference(args)
        
        if "error" in stats:
            logger.error(f"Inference failed: {stats['error']}")
            sys.exit(1)
        
        print(f"\n{'='*50}")
        print(f"Prediction completed successfully!")
        print(f"Results saved to: {stats['output_file']}")
        print(f"Successful SQL: {stats['successful_samples']}/{stats['total_samples']} samples")
        if stats['sorry_responses'] > 0:
            print(f"Sorry responses: {stats['sorry_responses']}")
        if stats['system_errors'] > 0:
            print(f"System errors: {stats['system_errors']}")
        print(f"Used {stats['workers']} concurrent workers")
        print(f"Throughput: {stats['throughput_samples_per_second']:.2f} samples/second")
        print(f"Total time: {stats['total_time_seconds']}s")
        print(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
