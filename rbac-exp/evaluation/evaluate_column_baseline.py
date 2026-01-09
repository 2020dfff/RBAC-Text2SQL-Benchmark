#!/usr/bin/env python3
"""
Baseline Text2SQL Evaluation (without RBAC).

Evaluates SQL generation accuracy without any RBAC-related metrics.
Used for comparing Text2SQL performance with/without RBAC constraints.

This module directly reads from the prediction _detailed.json file,
which already contains:
- The selected items (SystemManager role for baseline)
- gold_sql from the selected items
- Predictions and metadata

Metrics computed:
- Execution Accuracy (EX): percentage of queries that return correct results
- Per-difficulty breakdown (if applicable)

Output format aligned with experiments/eval/evaluation_spider.py.
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from local evaluation module (migrated from experiments)
from func_timeout import FunctionTimedOut, func_timeout

try:
    from evaluation.exec_eval import eval_exec_match
except ImportError:
    eval_exec_match = None

from configs.paths import get_dataset_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIMEOUT = 30
DIFFICULTY_LEVELS_SPIDER = ("easy", "medium", "hard", "extra")
DIFFICULTY_LEVELS_BIRD = ("simple", "moderate", "challenging")


def normalise_difficulty(raw: str, dataset: str) -> str:
    """Normalize difficulty level based on dataset."""
    value = (raw or "").strip().lower()
    
    if dataset.lower() == "spider":
        if value not in DIFFICULTY_LEVELS_SPIDER:
            return "extra"
        return value
    elif dataset.lower() == "bird":
        if value not in DIFFICULTY_LEVELS_BIRD:
            return "challenging"
        return value
    else:
        return value or "unknown"


def load_detailed_predictions(file_path: str) -> List[Dict[str, Any]]:
    """
    Load predictions from _detailed.json file.
    
    This file contains complete information for each prediction including:
    - id, database, gold_sql, is_allowed
    - pred_sql, raw_response
    - difficulty, mode, metadata
    """
    logger.info(f"Loading detailed predictions from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} predictions")
    return data


def exec_match_with_timeout(
    db_path: str,
    predicted: str,
    gold: str,
    plug_value: bool = False,
    keep_distinct: bool = False,
) -> bool:
    """Execute SQL with timeout."""
    if eval_exec_match is None:
        logger.warning("eval_exec_match not available, falling back to string comparison")
        return predicted.strip().lower() == gold.strip().lower()
    
    try:
        return bool(
            func_timeout(
                TIMEOUT,
                eval_exec_match,
                kwargs={
                    "db": db_path,
                    "p_str": predicted,
                    "g_str": gold,
                    "plug_value": plug_value,
                    "keep_distinct": keep_distinct,
                    "progress_bar_for_each_datapoint": False,
                },
            )
        )
    except FunctionTimedOut:
        return False
    except Exception as e:
        logger.debug(f"SQL execution error: {e}")
        return False


def evaluate_baseline(
    predict_file: str,
    dataset: str,
    db_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    plug_value: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate baseline Text2SQL predictions (without RBAC).
    
    This function directly uses the _detailed.json file from prediction,
    which already contains SystemManager role items with valid gold_sql.
    
    Args:
        predict_file: Path to predictions (_detailed.json)
        dataset: Dataset name (spider, bird, livesqlbench)
        db_dir: Path to database directory
        output_dir: Path to save evaluation results
        plug_value: Whether to try plugging gold values (slow!)
        
    Returns:
        Dictionary with evaluation results
    """
    # Setup paths
    if db_dir is None:
        paths = get_dataset_paths(dataset)
        db_dir = paths["db"]
    
    if output_dir is None:
        output_dir = os.path.join("rbac-exp", "output", "eval_result")
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle both .sql and _detailed.json files
    if predict_file.endswith('.sql'):
        detailed_file = predict_file.replace('.sql', '_detailed.json')
    else:
        detailed_file = predict_file
    
    if not os.path.exists(detailed_file):
        logger.error(f"Detailed prediction file not found: {detailed_file}")
        logger.error("Please run prediction with baseline mode first to generate _detailed.json file")
        return {"error": f"File not found: {detailed_file}"}
    
    # Extract base name for output files
    base = os.path.splitext(os.path.basename(predict_file))[0]
    base = base.replace("_detailed", "")
    
    # Load predictions from _detailed.json
    predictions = load_detailed_predictions(detailed_file)
    
    total = len(predictions)
    
    # Initialize buckets by difficulty
    if dataset.lower() == "spider":
        difficulty_levels = list(DIFFICULTY_LEVELS_SPIDER)
    elif dataset.lower() == "bird":
        difficulty_levels = list(DIFFICULTY_LEVELS_BIRD)
    else:
        difficulty_levels = []
    
    scores = {level: {"count": 0, "correct": 0, "wrong": 0, "empty": 0} for level in difficulty_levels}
    scores["all"] = {"count": 0, "correct": 0, "wrong": 0, "empty": 0}
    
    # Open incorrect log file
    incorrect_path = os.path.join(output_dir, f"{base}_baseline_incorrect.txt")
    incorrect_log = open(incorrect_path, "w", encoding="utf-8")
    
    logger.info(f"Starting baseline evaluation of {total} samples...")
    logger.info(f"Database directory: {db_dir}")
    
    # Track detailed results for JSON output
    detailed_results = []
    
    # Process each sample
    for idx, item in enumerate(predictions):
        # Extract item fields from _detailed.json
        sample_id = item.get("id", f"sample_{idx}")
        db_id = str(item.get("database", "")).strip()
        difficulty = normalise_difficulty(str(item.get("difficulty", "")), dataset)
        
        # Get gold SQL from the item (already correct for SystemManager)
        gold_sql = str(item.get("gold_sql", "")).strip()
        
        # Get prediction
        pred_sql = str(item.get("pred_sql", "")).strip()
        
        # Get metadata for logging
        metadata = item.get("metadata", {})
        question = metadata.get("question") or item.get("input") or ""
        if isinstance(question, dict):
            question = question.get("input", "")
        
        # Determine which buckets to update
        buckets = [scores[difficulty], scores["all"]] if difficulty in scores else [scores["all"]]
        
        # Increment count
        for bucket in buckets:
            bucket["count"] += 1
        
        # Skip if gold SQL is invalid (shouldn't happen with SystemManager)
        if not gold_sql or "sorry" in gold_sql.lower():
            logger.warning(f"Sample {idx}: Invalid gold SQL '{gold_sql[:50]}...', skipping")
            for bucket in buckets:
                bucket["count"] -= 1  # Don't count this sample
            continue
        
        # Record for detailed output
        result_entry = {
            "index": idx + 1,
            "id": sample_id,
            "db_id": db_id,
            "difficulty": difficulty,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "question": question[:200] if question else "",
        }
        
        # Check if prediction is empty
        if not pred_sql:
            for bucket in buckets:
                bucket["empty"] += 1
                bucket["wrong"] += 1
            result_entry["result"] = "empty"
            result_entry["exec_match"] = False
            detailed_results.append(result_entry)
            incorrect_log.write(
                f"index: {idx + 1}\n"
                f"db_id: {db_id}\n"
                f"question: {question[:200]}\n"
                f"{difficulty} pred: <empty>\n"
                f"{difficulty} gold: {gold_sql}\n\n"
            )
            continue
        
        # Check if prediction is a refusal (model incorrectly refused)
        if "sorry" in pred_sql.lower() and "cannot" in pred_sql.lower():
            for bucket in buckets:
                bucket["wrong"] += 1
            result_entry["result"] = "unnecessary_refusal"
            result_entry["exec_match"] = False
            detailed_results.append(result_entry)
            incorrect_log.write(
                f"index: {idx + 1}\n"
                f"db_id: {db_id}\n"
                f"result_type: unnecessary_refusal\n"
                f"{difficulty} pred: {pred_sql}\n"
                f"{difficulty} gold: {gold_sql}\n\n"
            )
            continue
        
        # Execute SQL for correctness
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite") if db_id else ""
        
        if db_path and os.path.exists(db_path):
            exec_correct = exec_match_with_timeout(db_path, pred_sql, gold_sql, plug_value=plug_value)
        else:
            # Fallback to string comparison
            logger.warning(f"Database not found: {db_path}, using string comparison")
            exec_correct = pred_sql.strip().lower() == gold_sql.strip().lower()
        
        result_entry["exec_match"] = exec_correct
        
        if exec_correct:
            for bucket in buckets:
                bucket["correct"] += 1
            result_entry["result"] = "correct"
        else:
            for bucket in buckets:
                bucket["wrong"] += 1
            result_entry["result"] = "wrong"
            incorrect_log.write(
                f"index: {idx + 1}\n"
                f"db_id: {db_id}\n"
                f"{difficulty} pred: {pred_sql}\n"
                f"{difficulty} gold: {gold_sql}\n\n"
            )
        
        detailed_results.append(result_entry)
        
        # Progress logging
        if (idx + 1) % 100 == 0:
            current_acc = scores["all"]["correct"] / scores["all"]["count"] * 100 if scores["all"]["count"] > 0 else 0
            logger.info(f"Evaluated {idx + 1}/{total} samples, current EX: {current_acc:.2f}%")
    
    incorrect_log.close()
    logger.info(f"Incorrect cases logged to: {incorrect_path}")
    
    # Compute metrics
    all_bucket = scores["all"]
    total_count = all_bucket["count"]
    
    if total_count > 0:
        ex_accuracy = all_bucket["correct"] / total_count * 100
    else:
        ex_accuracy = 0.0
    
    # Per-difficulty metrics
    per_diff_metrics = {}
    for level in difficulty_levels:
        bucket = scores[level]
        if bucket["count"] > 0:
            per_diff_metrics[level] = {
                "count": bucket["count"],
                "correct": bucket["correct"],
                "wrong": bucket["wrong"],
                "empty": bucket["empty"],
                "accuracy": bucket["correct"] / bucket["count"] * 100,
            }
    
    # Build result summary (aligned with experiments format)
    results = {
        "mode": "baseline",
        "dataset": dataset,
        "total_samples": total_count,
        "execution_accuracy": round(ex_accuracy, 2),
        "correct": all_bucket["correct"],
        "wrong": all_bucket["wrong"],
        "empty": all_bucket["empty"],
        "per_difficulty": per_diff_metrics,
        "detailed_results": detailed_results,
    }
    
    # Generate result string (aligned with experiments/eval output format)
    result_lines = [
        "=" * 70,
        "BASELINE TEXT2SQL EVALUATION RESULTS",
        "=" * 70,
        f"Dataset: {dataset}",
        f"Prediction file: {os.path.basename(predict_file)}",
        f"Total Samples: {total_count}",
        "",
        "-" * 70,
        "EXECUTION ACCURACY (EX)",
        "-" * 70,
    ]
    
    # Per-difficulty breakdown (formatted like experiments output)
    if per_diff_metrics:
        for level in difficulty_levels:
            if level in per_diff_metrics:
                m = per_diff_metrics[level]
                result_lines.append(f"  {level:12s}: {m['accuracy']:6.2f}% ({m['correct']:4d}/{m['count']:4d})")
    
    result_lines.append("-" * 70)
    result_lines.append(f"  {'all':12s}: {ex_accuracy:6.2f}% ({all_bucket['correct']:4d}/{total_count:4d})")
    result_lines.append("=" * 70)
    
    # Summary statistics
    result_lines.extend([
        "",
        "Summary:",
        f"  Correct: {all_bucket['correct']}",
        f"  Wrong: {all_bucket['wrong']}",
        f"  Empty: {all_bucket['empty']}",
        "",
        f"Overall Execution Accuracy (EX): {ex_accuracy:.2f}%",
        "=" * 70,
    ])
    
    result_str = "\n".join(result_lines)
    print(result_str)
    
    # Save results
    result_path = os.path.join(output_dir, f"{base}_baseline_evaluate_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result_str)
    logger.info(f"Results saved to: {result_path}")
    
    # Save JSON results (without detailed_results for smaller file)
    json_results = {k: v for k, v in results.items() if k != "detailed_results"}
    json_path = os.path.join(output_dir, f"{base}_baseline_evaluate_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"JSON results saved to: {json_path}")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Baseline Text2SQL Evaluation (without RBAC)")
    
    parser.add_argument("--pred", type=str, required=True,
                        help="Path to prediction file (.sql or _detailed.json)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["spider", "bird", "livesqlbench"],
                        help="Dataset name")
    parser.add_argument("--db_dir", type=str, default=None,
                        help="Path to database directory (optional, auto-detected)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--plug_value", action="store_true",
                        help="Enable value plugging (slow!)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    evaluate_baseline(
        predict_file=args.pred,
        dataset=args.dataset,
        db_dir=args.db_dir,
        output_dir=args.output_dir,
        plug_value=args.plug_value,
    )


if __name__ == "__main__":
    main()
