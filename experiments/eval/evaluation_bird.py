"""
do evaluate about the predict sql in dataset BIRD,compare with default dev.sql
--db
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sqlite3
import sys
import time

from func_timeout import FunctionTimedOut, func_timeout

from livesqlbench_utils import (
    compute_access_control_metrics,
    compute_answerable,
    compute_exec_accuracy,
    compute_sql_metrics,
    init_bucket,
    is_refusal,
)

DIFFICULTY_LEVELS = ["simple", "moderate", "challenging", "unknown"]


def load_json(dir):
    with open(dir, "r") as j:
        contents = json.loads(j.read())
    return contents

def load_difficulty(diff_file_path):
    """Load difficulty information from text file or JSON file"""
    if not diff_file_path or not os.path.exists(diff_file_path):
        return []
    
    try:
        # Try JSON format first
        with open(diff_file_path, "r") as f:
            contents = json.loads(f.read())
        return contents
    except json.JSONDecodeError:
        # Fall back to text format (one difficulty per line)
        with open(diff_file_path, "r") as f:
            lines = f.readlines()
        return [{"difficulty": line.strip()} for line in lines if line.strip()]


def result_callback(result):
    exec_result.append(result)


def execute_sql(predicted_sql, ground_truth, db_path):
    """Optimized SQL execution with connection reuse and early termination"""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        conn.execute("PRAGMA cache_size=10000")  # Larger cache
        cursor = conn.cursor()
        
        # Quick validation: skip obviously invalid queries
        if not predicted_sql.strip() or 'Sorry' in predicted_sql:
            return 0, 0
            
        pred_start_time = time.time()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        pred_exec_time = time.time() - pred_start_time
        
        true_start_time = time.time()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        true_exec_time = time.time() - true_start_time
        
        res = 0
        time_ratio = 0
        if set(predicted_res) == set(ground_truth_res):
            res = 1
            time_ratio = true_exec_time / pred_exec_time if pred_exec_time > 0 else 0
        return res, time_ratio
    finally:
        if conn:
            conn.close()


def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    """Optimized model execution with faster timeout and better error handling"""
    try:
        # Quick pre-check for obviously invalid queries
        if not predicted_sql.strip() or len(predicted_sql) > 10000:
            res, time_ratio = 0, 0
        else:
            res, time_ratio = func_timeout(
                meta_time_out, execute_sql, args=(predicted_sql, ground_truth, db_place)
            )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = 0
        time_ratio = 0
    except Exception as e:
        # Log specific error types for debugging
        error_type = type(e).__name__
        if "syntax" in str(e).lower() or "near" in str(e).lower():
            pass  # SQL syntax error - expected
        res = 0
        time_ratio = 0
        
    result = {
        "sql_idx": idx,
        "res": res,
        "match": int(predicted_sql == ground_truth),
        "time_ratio": time_ratio,
    }
    return result


def package_sqls(sql_path, db_root_path, mode="gpt", data_mode="dev"):
    clean_sqls = []
    db_path_list = []
    if mode == "gpt":
        # For prediction files, we need to get db_id from bird_dev.json to match each prediction
        bird_json_path = "./data/selected/bird/bird_dev_data.json"
        try:
            with open(bird_json_path, 'r') as f:
                bird_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find {bird_json_path}, using default database names")
            bird_data = []
        
        with open(sql_path) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                clean_sqls.append(line.strip())
                
                # Get db_id from bird_dev.json if available
                if idx < len(bird_data):
                    db_name = bird_data[idx]['db_id']
                else:
                    db_name = "financial"  # fallback
                
                db_path_list.append(db_root_path + db_name + "/" + db_name + ".sqlite")
    elif mode == "gt":
        sqls = open(sql_path)
        sql_txt = sqls.readlines()
        for idx, sql_str in enumerate(sql_txt):
            # bird_dev_gold.txt format: SQL\tdb_name
            parts = sql_str.strip().split("\t")
            if len(parts) >= 2:
                sql, db_name = parts[0], parts[1]
            else:
                sql, db_name = parts[0], "financial"  # fallback
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + "/" + db_name + ".sqlite")

    return clean_sqls, db_path_list


def is_likely_valid_sql(sql):
    """Quick pre-check to identify obviously invalid SQL queries"""
    if not sql or not sql.strip():
        return False
    if 'Sorry' in sql or 'Error:' in sql:
        return False
    if len(sql) > 5000:  # Very long queries are often problematic
        return False
    sql_upper = sql.upper().strip()
    # Must start with a valid SQL command
    if not any(sql_upper.startswith(cmd) for cmd in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']):
        return False
    return True

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    """Optimized parallel SQL execution with pre-filtering"""
    print(f"Starting parallel evaluation with {num_cpus} processes, timeout={meta_time_out}s")
    
    # Pre-filter obviously invalid queries
    valid_indices = []
    for i, (predicted_sql, ground_truth) in enumerate(sqls):
        if is_likely_valid_sql(predicted_sql) and is_likely_valid_sql(ground_truth):
            valid_indices.append(i)
        else:
            # Add failed result directly
            exec_result.append({
                "sql_idx": i,
                "res": 0,
                "match": int(predicted_sql == ground_truth),
                "time_ratio": 0,
            })
    
    print(f"Pre-filtered: {len(valid_indices)} valid queries out of {len(sqls)} total")
    
    if not valid_indices:
        return
        
    pool = mp.Pool(processes=num_cpus)
    for i in valid_indices:
        predicted_sql, ground_truth = sqls[i]
        pool.apply_async(
            execute_model,
            args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out),
            callback=result_callback,
        )
    pool.close()
    pool.join()


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x["sql_idx"])


def compute_ves(exec_results):
    num_queries = len(exec_results)
    total_ratio = 0
    count = 0

    for i, result in enumerate(exec_results):
        if result["time_ratio"] != 0:
            count += 1
        total_ratio += math.sqrt(result["time_ratio"]) * 100
    ves = total_ratio / num_queries
    return ves


def compute_acc_by_diff(exec_results, difficulty_contents, metric):
    num_queries = len(exec_results)
    results = [res[metric] for res in exec_results]
    contents = difficulty_contents
    simple_results, moderate_results, challenging_results = [], [], []

    # Handle case where difficulty info is missing or length mismatch
    if not contents or len(contents) != num_queries:
        print(f"Warning: Difficulty info missing or length mismatch. "
              f"Expected {num_queries}, got {len(contents) if contents else 0}. "
              f"Treating all queries as 'simple'.")
        # Treat all queries as simple
        simple_results = exec_results
        moderate_results = []
        challenging_results = []
    else:
        for i, content in enumerate(contents):
            # Handle both dict format and simple string format from bird_difficulty.txt
            if isinstance(content, dict):
                difficulty = content["difficulty"]
            else:
                difficulty = content  # Direct string from bird_difficulty.txt
                
            if difficulty == "simple":
                simple_results.append(exec_results[i])
            elif difficulty == "moderate":
                moderate_results.append(exec_results[i])
            elif difficulty == "challenging":
                challenging_results.append(exec_results[i])
    if metric in ["res", "match"]:
        simple_acc = sum([res[metric] for res in simple_results]) / len(simple_results) if simple_results else 0
        moderate_acc = sum([res[metric] for res in moderate_results]) / len(moderate_results) if moderate_results else 0
        challenging_acc = sum([res[metric] for res in challenging_results]) / len(challenging_results) if challenging_results else 0
        all_acc = sum(results) / num_queries
    elif metric in ["time_ratio"]:
        simple_acc = compute_ves(simple_results) if simple_results else 0
        moderate_acc = compute_ves(moderate_results) if moderate_results else 0
        challenging_acc = compute_ves(challenging_results) if challenging_results else 0
        all_acc = compute_ves(exec_results)
    else:
        raise NotImplementedError(f"metric: {metric} is not supported")
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        num_queries,
    ]
    if metric in ["res", "match"]:
        return (
            simple_acc * 100,
            moderate_acc * 100,
            challenging_acc * 100,
            all_acc * 100,
            count_lists,
        )
    else:
        return simple_acc, moderate_acc, challenging_acc, all_acc, count_lists


def print_data(score_lists, count_lists, metric="Exec ACCURACY"):
    levels = ["simple", "moderate", "challenging", "total"]
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format("count", *count_lists))

    print(
        f"====================================== {metric} ====================================="
    )
    print(
        "{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format("accuracy", *score_lists)
    )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--predicted_sql_path",
        type=str,
        default="./experiments/output/pred/pred_deepseek-coder_bird_cleaned.sql",
    )
    args_parser.add_argument(
        "--ground_truth_path",
        type=str,
        default="./data/selected/gold_data/bird_dev_gold.txt",
    )
    args_parser.add_argument("--data_mode", type=str, default="dev")
    args_parser.add_argument(
        "--db_root_path",
        type=str,
        default="./data/Bird/dev_20240627/dev_databases/",
    )
    args_parser.add_argument("--num_cpus", type=int, default=1)
    args_parser.add_argument("--meta_time_out", type=float, default=10.0, help="Timeout for each query (seconds)")
    args_parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (for quick testing)")
    args_parser.add_argument("--skip_timeout_queries", action="store_true", help="Skip queries that are likely to timeout")
    args_parser.add_argument("--mode_gt", type=str, default="gt")
    args_parser.add_argument("--mode_predict", type=str, default="gpt")
    args_parser.add_argument("--difficulty", type=str, default="simple")
    args_parser.add_argument("--diff_json_path", type=str, default="./data/selected/gold_data/bird_difficulty.txt")
    args_parser.add_argument("--fair_comparison", action='store_true', help="Enable fair comparison mode")
    args_parser.add_argument(
        "--etype",
        dest="etype",
        type=str,
        default="match",
        choices=("all", "exec", "match", "ves"),
    )

    args = args_parser.parse_args()
    exec_result = []

    # Setup output files with spider-compatible naming
    base = re.sub(r"\.sql$|\.jsonl$|\.txt$", "", os.path.basename(args.predicted_sql_path))
    suffix = "_fair_comparison" if args.fair_comparison else ""
    
    # Create eval_result directory structure like spider
    eval_result_dir = "experiments/output/eval_result"
    os.makedirs(eval_result_dir, exist_ok=True)
    
    # Create incorrect cases log file
    incorrect_filename = os.path.join(eval_result_dir, f"{base}{suffix}_incorrect.txt")
    incorrect_log_file = open(incorrect_filename, "w")
    
    # Create evaluation result file
    result_filename = os.path.join(eval_result_dir, f"{base}{suffix}_evaluate_result.txt")

    pred_queries, db_paths = package_sqls(
        args.predicted_sql_path,
        args.db_root_path,
        mode=args.mode_predict,
        data_mode=args.data_mode,
    )
    # generate gt sqls:
    gt_queries, db_paths_gt = package_sqls(
        args.ground_truth_path, args.db_root_path, mode="gt", data_mode=args.data_mode
    )

    if len(db_paths) == 0:
        db_paths = db_paths_gt

    query_pairs = list(zip(pred_queries, gt_queries))
    
    # Load difficulty information for logging incorrect cases (before sample limiting)
    difficulty_info = load_difficulty(args.diff_json_path)
    
    # Apply sample limit for quick testing
    if args.max_samples and args.max_samples < len(query_pairs):
        print(f"Quick evaluation mode: Testing first {args.max_samples} samples out of {len(query_pairs)}")
        query_pairs = query_pairs[:args.max_samples]
        db_paths = db_paths[:args.max_samples]
        # Also limit difficulty info to match
        if difficulty_info:
            difficulty_info = difficulty_info[:args.max_samples]
    
    # Filter out likely timeout queries if requested
    if args.skip_timeout_queries:
        filtered_pairs = []
        filtered_paths = []
        for i, (pred, gt) in enumerate(query_pairs):
            # Skip very long queries that might timeout
            if len(pred) < 1000 and len(gt) < 1000 and 'UNION' not in pred.upper() and 'JOIN' not in pred.upper()[:100]:
                filtered_pairs.append((pred, gt))
                filtered_paths.append(db_paths[i])
        print(f"Filtered from {len(query_pairs)} to {len(filtered_pairs)} queries (skipped complex ones)")
        query_pairs = filtered_pairs
        db_paths = filtered_paths
    
    if args.etype in ["all", "exec", "ves"]:
        run_sqls_parallel(
            query_pairs,
            db_places=db_paths,
            num_cpus=args.num_cpus,
            meta_time_out=args.meta_time_out,
        )
    else:
        for i, sql_pair in enumerate(query_pairs):
            predicted_sql, ground_truth = sql_pair
            exec_result.append(
                {"sql_idx": i, "match": int(predicted_sql == ground_truth)}
            )
    exec_result = sort_results(exec_result)
    
    bucket = init_bucket()
    difficulty_buckets = {level: init_bucket() for level in DIFFICULTY_LEVELS}
    exec_result_map = {item.get("sql_idx"): item for item in exec_result}

    for idx, (pred_sql, gold_sql) in enumerate(query_pairs):
        entry = exec_result_map.get(idx, {})
        res_value = entry.get("res")
        if res_value is None:
            res_value = entry.get("match")
        res_value = 1 if res_value else 0

        pred_text = pred_sql or ""
        gold_text = gold_sql or ""

        predicted_refusal = bool(not pred_text.strip() or is_refusal(pred_text))
        should_refuse = bool(not gold_text.strip() or is_refusal(gold_text))

        difficulty_label = "unknown"
        if difficulty_info and idx < len(difficulty_info):
            diff_entry = difficulty_info[idx]
            if isinstance(diff_entry, dict):
                difficulty_label = str(diff_entry.get("difficulty", "unknown")).strip().lower() or "unknown"
            else:
                difficulty_label = str(diff_entry).strip().lower() or "unknown"
        if difficulty_label not in difficulty_buckets:
            difficulty_buckets[difficulty_label] = init_bucket()
        target_buckets = [bucket, difficulty_buckets[difficulty_label]]

        for target in target_buckets:
            target["count"] += 1

        if should_refuse:
            if predicted_refusal:
                for target in target_buckets:
                    target["correct_refusal"] += 1
            else:
                if res_value:
                    for target in target_buckets:
                        target["violation_correct"] += 1
                else:
                    for target in target_buckets:
                        target["violation_wrong"] += 1
        else:
            if predicted_refusal:
                for target in target_buckets:
                    target["incorrect_refusal"] += 1
            else:
                if res_value:
                    for target in target_buckets:
                        target["correct"] += 1
                else:
                    for target in target_buckets:
                        target["wrong"] += 1

    access_control_metrics = compute_access_control_metrics(bucket)
    sql_metrics = compute_sql_metrics(bucket)
    exec_accuracy_overall = compute_exec_accuracy(bucket)
    answerable_total = compute_answerable(bucket)
    per_difficulty_metrics = {}
    for level, diff_bucket in difficulty_buckets.items():
        if diff_bucket["count"]:
            per_difficulty_metrics[level] = {
                "bucket": diff_bucket,
                "access": compute_access_control_metrics(diff_bucket),
                "sql": compute_sql_metrics(diff_bucket),
                "exec_accuracy": compute_exec_accuracy(diff_bucket),
                "answerable": compute_answerable(diff_bucket),
            }

    print("start calculate")
    
    # Log incorrect cases
    if args.etype in ["all", "exec"]:
        for result in exec_result:
            idx = result["sql_idx"]
            if result.get("res", 0) == 0:  # Incorrect execution result (changed from "0" to 0)
                if difficulty_info and idx < len(difficulty_info):
                    diff_entry = difficulty_info[idx]
                    difficulty = diff_entry.get("difficulty") if isinstance(diff_entry, dict) else diff_entry
                else:
                    difficulty = "unknown"
                predicted_sql, ground_truth = query_pairs[idx]
                # Get database ID from path
                db_path = db_paths[idx] if idx < len(db_paths) else ""
                db_id = os.path.basename(db_path).replace(".sqlite", "") if db_path else "unknown"
                
                incorrect_log_file.write(f"index: {idx+1}\n")
                incorrect_log_file.write(f"db_id: {db_id}\n")
                incorrect_log_file.write(f"difficulty: {difficulty}\n")
                incorrect_log_file.write(f"pred: {predicted_sql}\n")
                incorrect_log_file.write(f"gold: {ground_truth}\n\n")
    elif args.etype in ["all", "match"]:
        for result in exec_result:
            idx = result["sql_idx"]
            if result.get("match", 0) == 0:  # Incorrect match result
                if difficulty_info and idx < len(difficulty_info):
                    diff_entry = difficulty_info[idx]
                    difficulty = diff_entry.get("difficulty") if isinstance(diff_entry, dict) else diff_entry
                else:
                    difficulty = "unknown"
                predicted_sql, ground_truth = query_pairs[idx]
                db_path = db_paths[idx] if idx < len(db_paths) else ""
                db_id = os.path.basename(db_path).replace(".sqlite", "") if db_path else "unknown"
                
                incorrect_log_file.write(f"index: {idx+1}\n")
                incorrect_log_file.write(f"db_id: {db_id}\n")
                incorrect_log_file.write(f"difficulty: {difficulty}\n")
                incorrect_log_file.write(f"pred: {predicted_sql}\n")
                incorrect_log_file.write(f"gold: {ground_truth}\n\n")
    
    incorrect_log_file.close()
    
    # Generate formatted evaluation results
    with open(result_filename, "w") as result_file:
        if args.etype in ["all", "exec"]:
            (
                simple_acc,
                moderate_acc,
                challenging_acc,
                acc,
                count_lists,
            ) = compute_acc_by_diff(exec_result, difficulty_info, "res")
            score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
            print_data(score_lists, count_lists, metric="Exec Accuracy")
            
            # Write to result file
            result_file.write("{:<20} {:<20} {:<20} {:<20} {:<20}\n".format(
                "simple", "moderate", "challenging", "extra", "all"
            ))
            result_file.write("{:<20} {:<20} {:<20} {:<20} {:<20}\n".format(
                "count", *count_lists
            ))
            result_file.write("compare etype exec\n")
            result_file.write("=====================   EXECUTION ACCURACY     =====================\n")
            result_file.write("{:<20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}\n".format(
                "execution", *score_lists
            ))
            
        if args.etype in ["all", "match"]:
            (
                simple_acc,
                moderate_acc,
                challenging_acc,
                acc,
                count_lists,
            ) = compute_acc_by_diff(exec_result, difficulty_info, "match")
            score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
            print_data(score_lists, count_lists, metric="Match Accuracy")
            
            # Write to result file
            if args.etype == "match":  # Only write header if not already written
                result_file.write("{:<20} {:<20} {:<20} {:<20} {:<20}\n".format(
                    "simple", "moderate", "challenging", "extra", "all"
                ))
                result_file.write("{:<20} {:<20} {:<20} {:<20} {:<20}\n".format(
                    "count", *count_lists
                ))
            result_file.write("compare etype match\n")
            result_file.write("=====================   MATCH ACCURACY     =====================\n")
            result_file.write("{:<20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}\n".format(
                "match", *score_lists
            ))
            
        if args.etype in ["all", "ves"]:
            (
                simple_acc,
                moderate_acc,
                challenging_acc,
                acc,
                count_lists,
            ) = compute_acc_by_diff(exec_result, difficulty_info, "time_ratio")
            score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
            print_data(score_lists, count_lists, metric="Ves")
            
            # Write to result file
            if args.etype == "ves":  # Only write header if not already written
                result_file.write("{:<20} {:<20} {:<20} {:<20} {:<20}\n".format(
                    "simple", "moderate", "challenging", "extra", "all"
                ))
                result_file.write("{:<20} {:<20} {:<20} {:<20} {:<20}\n".format(
                    "count", *count_lists
                ))
            result_file.write("compare etype ves\n")
            result_file.write("=====================   VES ACCURACY     =====================\n")
            result_file.write("{:<20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}\n".format(
                "ves", *score_lists
            ))

        result_file.write("\nACCESS CONTROL & SQL PERFORMANCE SUMMARY\n")
        result_file.write("----------------------------------------\n")
        result_file.write(f"Total samples: {int(bucket['count'])}\n")
        result_file.write(f"Answerable samples: {int(answerable_total)}\n")
        result_file.write(f"Execution accuracy: {exec_accuracy_overall:.4f}\n")
        result_file.write(
            "Six-category counts (C/W/CR/IR/VC/VW): "
            f"{int(bucket['correct'])} / {int(bucket['wrong'])} / {int(bucket['correct_refusal'])} / "
            f"{int(bucket['incorrect_refusal'])} / {int(bucket['violation_correct'])} / {int(bucket['violation_wrong'])}\n"
        )
        result_file.write(
            "Access Control -> TP/FP/FN/TN: "
            f"{access_control_metrics['tp']} / {access_control_metrics['fp']} / "
            f"{access_control_metrics['fn']} / {access_control_metrics['tn']}\n"
        )
        result_file.write(
            "Access Control metrics: Precision={:.4f}, Recall={:.4f}, F1={:.4f}, Accuracy={:.4f}, ViolationRate={:.4f}, OverRefusalRate={:.4f}\n".format(
                access_control_metrics["precision"],
                access_control_metrics["recall"],
                access_control_metrics["f1"],
                access_control_metrics["accuracy"],
                access_control_metrics["violation_rate"],
                access_control_metrics["over_refusal_rate"],
            )
        )
        result_file.write(
            "SQL performance: SQL-emitting={} | CorrectSQL={} | SQLAccuracy={:.4f} | SafeSQLAccuracy={:.4f} | SafeEX={:.4f} | UnsafeEX={:.4f}\n".format(
                sql_metrics["sql_attempts"],
                sql_metrics["correct_sql"],
                sql_metrics["sql_accuracy"],
                sql_metrics["safe_sql_accuracy"],
                sql_metrics["safe_ex"],
                sql_metrics["unsafe_ex"],
            )
        )

        if per_difficulty_metrics:
            result_file.write("\nPER-DIFFICULTY ACCESS CONTROL & SQL METRICS\n")
            result_file.write("-------------------------------------------\n")
            for level in DIFFICULTY_LEVELS:
                metrics = per_difficulty_metrics.get(level)
                if not metrics:
                    continue
                diff_bucket = metrics["bucket"]
                access = metrics["access"]
                sql = metrics["sql"]
                result_file.write(
                    f"[{level}] count={int(diff_bucket['count'])} answerable={int(metrics['answerable'])} exec_acc={metrics['exec_accuracy']:.4f}\n"
                )
                result_file.write(
                    "  Six-category (C/W/CR/IR/VC/VW): {} / {} / {} / {} / {} / {}\n".format(
                        int(diff_bucket["correct"]),
                        int(diff_bucket["wrong"]),
                        int(diff_bucket["correct_refusal"]),
                        int(diff_bucket["incorrect_refusal"]),
                        int(diff_bucket["violation_correct"]),
                        int(diff_bucket["violation_wrong"]),
                    )
                )
                result_file.write(
                    "  Access Control -> TP/FP/FN/TN: {} / {} / {} / {} | Precision={:.4f} Recall={:.4f} F1={:.4f} Acc={:.4f} | ViolationRate={:.4f} OverRefusalRate={:.4f}\n".format(
                        access["tp"],
                        access["fp"],
                        access["fn"],
                        access["tn"],
                        access["precision"],
                        access["recall"],
                        access["f1"],
                        access["accuracy"],
                        access["violation_rate"],
                        access["over_refusal_rate"],
                    )
                )
                result_file.write(
                    "  SQL performance: attempts={} correct={} SQLAcc={:.4f} SafeSQLAcc={:.4f} SafeEX={:.4f} UnsafeEX={:.4f}\n".format(
                        sql["sql_attempts"],
                        sql["correct_sql"],
                        sql["sql_accuracy"],
                        sql["safe_sql_accuracy"],
                        sql["safe_ex"],
                        sql["unsafe_ex"],
                    )
                )
    print(
        "==========================================================================================="
    )
    print("Finished evaluation")
    print(f"Evaluation results saved to: {result_filename}")
    print(f"Incorrect cases logged to: {incorrect_filename}")

    print("\nAccess Control & SQL Performance Summary")
    print("----------------------------------------")
    print(f"Total samples           : {int(bucket['count'])}")
    print(f"Answerable samples      : {int(answerable_total)}")
    print(f"Execution accuracy      : {exec_accuracy_overall:.4f}")
    print(
        "Six-category counts (C/W/CR/IR/VC/VW): "
        f"{int(bucket['correct'])} / {int(bucket['wrong'])} / {int(bucket['correct_refusal'])} / "
        f"{int(bucket['incorrect_refusal'])} / {int(bucket['violation_correct'])} / {int(bucket['violation_wrong'])}"
    )
    print(
        "Access Control -> TP/FP/FN/TN: "
        f"{access_control_metrics['tp']} / {access_control_metrics['fp']} / "
        f"{access_control_metrics['fn']} / {access_control_metrics['tn']}"
    )
    print(
        "Precision / Recall / F1 / Accuracy : {:.4f} / {:.4f} / {:.4f} / {:.4f}".format(
            access_control_metrics["precision"],
            access_control_metrics["recall"],
            access_control_metrics["f1"],
            access_control_metrics["accuracy"],
        )
    )
    print(
        "Violation rate / Over-refusal rate : {:.4f} / {:.4f}".format(
            access_control_metrics["violation_rate"],
            access_control_metrics["over_refusal_rate"],
        )
    )
    print(
        "SQL metrics -> SQL-emitting={} | CorrectSQL={} | SQLAccuracy={:.4f} | SafeSQLAccuracy={:.4f} | SafeEX={:.4f} | UnsafeEX={:.4f}".format(
            sql_metrics["sql_attempts"],
            sql_metrics["correct_sql"],
            sql_metrics["sql_accuracy"],
            sql_metrics["safe_sql_accuracy"],
            sql_metrics["safe_ex"],
            sql_metrics["unsafe_ex"],
        )
    )

    if per_difficulty_metrics:
        print("\nPer-difficulty Access Control & SQL Metrics")
        print("-------------------------------------------")
        for level in DIFFICULTY_LEVELS:
            metrics = per_difficulty_metrics.get(level)
            if not metrics:
                continue
            diff_bucket = metrics["bucket"]
            access = metrics["access"]
            sql = metrics["sql"]
            print(
                f"[{level}] count={int(diff_bucket['count'])} answerable={int(metrics['answerable'])} exec_acc={metrics['exec_accuracy']:.4f}"
            )
            print(
                "  Six-category (C/W/CR/IR/VC/VW): {} / {} / {} / {} / {} / {}".format(
                    int(diff_bucket["correct"]),
                    int(diff_bucket["wrong"]),
                    int(diff_bucket["correct_refusal"]),
                    int(diff_bucket["incorrect_refusal"]),
                    int(diff_bucket["violation_correct"]),
                    int(diff_bucket["violation_wrong"]),
                )
            )
            print(
                "  Access Control -> TP/FP/FN/TN: {} / {} / {} / {} | Precision={:.4f} Recall={:.4f} F1={:.4f} Acc={:.4f} | ViolationRate={:.4f} OverRefusalRate={:.4f}".format(
                    access["tp"],
                    access["fp"],
                    access["fn"],
                    access["tn"],
                    access["precision"],
                    access["recall"],
                    access["f1"],
                    access["accuracy"],
                    access["violation_rate"],
                    access["over_refusal_rate"],
                )
            )
            print(
                "  SQL performance: attempts={} correct={} SQLAcc={:.4f} SafeSQLAcc={:.4f} SafeEX={:.4f} UnsafeEX={:.4f}".format(
                    sql["sql_attempts"],
                    sql["correct_sql"],
                    sql["sql_accuracy"],
                    sql["safe_sql_accuracy"],
                    sql["safe_ex"],
                    sql["unsafe_ex"],
                )
            )
