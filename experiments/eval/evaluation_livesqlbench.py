#!/usr/bin/env python3
"""Evaluation script for LiveSQLBench predictions.

This evaluator mirrors the behaviour of the existing Spider/BIRD tooling
while aligning with LiveSQLBench's official verification logic:

* Query tasks are validated by executing both prediction and gold SQL on
  isolated copies of the target database and comparing result sets.
* Management tasks (those that modify database state) are verified by
  running the official embedded Python ``test_case`` functions after the
  prediction has been applied. Test cases are mandatory for management
  tasks; if they are missing the evaluator will flag the item.
* Refusals ("Sorry, I cannot answer.") are tracked so their correctness
  can be reported separately.

Usage example::

    python dbgpt-hub-sql/eval/evaluation_livesqlbench.py \
        --predictions outputs/livesql_data/preds.jsonl \
        --base-data data/livesqlbench-base-lite-sqlite/livesqlbench_data_sqlite.jsonl \
        --gold-sql data/livesqlbench-base-lite-sqlite/livesqlbench_sqlite_gt_kg_testcases_0528.jsonl \
        --db-root data/livesqlbench-base-lite-sqlite
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from livesqlbench_utils import (
    EvaluationError,
    GoldRecord,
    init_bucket,
    compute_access_control_metrics,
    compute_answerable,
    compute_exec_accuracy,
    compute_sql_metrics,
    compute_violation_rate,
    SQLExecutionError,
    TestCaseAssertionError,
    TimeoutExpired,
    extract_sql_snippet,
    evaluate_management_statements,
    evaluate_query_statements,
    is_refusal,
    load_livesql_gold,
    load_prediction_records,
    pretty_difficulty,
    split_sql_statements,
)

DIFFICULTY_LEVELS = ["simple", "moderate", "challenging", "unknown"]


def _ensure_output_dir(base_path: Path) -> None:
    base_path.mkdir(parents=True, exist_ok=True)


def evaluate_predictions(
    predictions_path: Path,
    base_data_path: Path,
    gold_sql_path: Path,
    db_root: Path,
    timeout: float,
    max_samples: int | None,
) -> Tuple[Dict[str, Dict[str, float]], List[str], List[str]]:
    predictions = load_prediction_records(predictions_path)
    gold_map = load_livesql_gold(base_data_path, gold_sql_path)

    ordered_records: List[GoldRecord] = list(gold_map.values())
    ordered_records.sort(key=lambda rec: rec.instance_id)
    if max_samples is not None:
        ordered_records = ordered_records[:max_samples]

    stats: Dict[str, Dict[str, float]] = {level: init_bucket() for level in DIFFICULTY_LEVELS}
    stats["all"] = init_bucket()

    incorrect_logs: List[str] = []
    skipped_instances: List[str] = []

    for record in ordered_records:
        difficulty = pretty_difficulty(record.difficulty)
        if difficulty not in stats:
            stats[difficulty] = init_bucket()

        prediction = predictions.get(record.instance_id)
        if prediction is None:
            stats[difficulty]["missing"] += 1
            stats["all"]["missing"] += 1
            skipped_instances.append(record.instance_id)
            continue

        buckets = [stats[difficulty], stats["all"]]

        for bucket in buckets:
            bucket["count"] += 1

        raw_prediction = prediction.get("prediction_text", "")
        should_refuse = not record.sql_statements and not record.test_cases
        predicted_refusal = is_refusal(raw_prediction)

        if predicted_refusal:
            if should_refuse:
                for bucket in buckets:
                    bucket["correct_refusal"] += 1
                continue
            for bucket in buckets:
                bucket["incorrect_refusal"] += 1
            incorrect_logs.append(
                f"instance_id: {record.instance_id}\n"
                f"db_id: {record.db_id}\n"
                f"reason: refusal but SQL expected\n"
                f"prediction: {raw_prediction.strip()}\n"
                "---\n"
            )
            continue

        if should_refuse:
            for bucket in buckets:
                bucket["violation_wrong"] += 1
            incorrect_logs.append(
                f"instance_id: {record.instance_id}\n"
                f"db_id: {record.db_id}\n"
                "reason: SQL provided but task expects refusal\n"
                f"prediction: {raw_prediction.strip()}\n"
                "---\n"
            )
            continue

        sql_text = extract_sql_snippet(raw_prediction)
        statements = split_sql_statements(sql_text)
        if not statements:
            for bucket in buckets:
                bucket["empty"] += 1
                bucket["wrong"] += 1
            incorrect_logs.append(
                f"instance_id: {record.instance_id}\n"
                f"db_id: {record.db_id}\n"
                "reason: no executable SQL statements detected\n"
                f"raw_prediction: {raw_prediction.strip()}\n"
                "---\n"
            )
            continue

        try:
            if record.is_management:
                evaluate_management_statements(record, statements, db_root, timeout)
                success = True
            else:
                success, pred_rows, gold_rows = evaluate_query_statements(record, statements, db_root, timeout)
        except TimeoutExpired as exc:
            for bucket in buckets:
                bucket["timeout"] += 1
                if should_refuse:
                    bucket["violation_wrong"] += 1
                else:
                    bucket["wrong"] += 1
            incorrect_logs.append(
                f"instance_id: {record.instance_id}\n"
                f"db_id: {record.db_id}\n"
                f"reason: timeout ({exc})\n"
                f"prediction: {raw_prediction.strip()}\n"
                "---\n"
            )
            continue
        except TestCaseAssertionError as exc:
            for bucket in buckets:
                bucket["testcase_failed"] += 1
                if should_refuse:
                    bucket["violation_wrong"] += 1
                else:
                    bucket["wrong"] += 1
            incorrect_logs.append(
                f"instance_id: {record.instance_id}\n"
                f"db_id: {record.db_id}\n"
                f"reason: test case failed ({exc})\n"
                f"prediction: {raw_prediction.strip()}\n"
                "---\n"
            )
            continue
        except SQLExecutionError as exc:
            for bucket in buckets:
                bucket["sql_error"] += 1
                if should_refuse:
                    bucket["violation_wrong"] += 1
                else:
                    bucket["wrong"] += 1
            incorrect_logs.append(
                f"instance_id: {record.instance_id}\n"
                f"db_id: {record.db_id}\n"
                f"reason: SQL execution error ({exc})\n"
                f"prediction: {raw_prediction.strip()}\n"
                "---\n"
            )
            continue
        except EvaluationError as exc:
            for bucket in buckets:
                if should_refuse:
                    bucket["violation_wrong"] += 1
                else:
                    bucket["wrong"] += 1
            incorrect_logs.append(
                f"instance_id: {record.instance_id}\n"
                f"db_id: {record.db_id}\n"
                f"reason: evaluation error ({exc})\n"
                f"prediction: {raw_prediction.strip()}\n"
                "---\n"
            )
            continue

        if record.is_management:
            success = True
            pred_rows = gold_rows = []  # placeholders for summary logging

        if success:
            for bucket in buckets:
                if should_refuse:
                    bucket["violation_correct"] += 1
                else:
                    bucket["correct"] += 1
        else:
            for bucket in buckets:
                if should_refuse:
                    bucket["violation_wrong"] += 1
                else:
                    bucket["wrong"] += 1
            incorrect_logs.append(
                f"instance_id: {record.instance_id}\n"
                f"db_id: {record.db_id}\n"
                "reason: result mismatch\n"
                f"prediction_rows: {pred_rows}\n"
                f"gold_rows: {gold_rows}\n"
                f"prediction: {raw_prediction.strip()}\n"
                "---\n"
            )

    return stats, incorrect_logs, skipped_instances


def print_and_save_summary(
    stats: Dict[str, Dict[str, float]],
    output_path: Path,
    *,
    skipped_instances: Optional[Sequence[str]] = None,
) -> None:
    lines: List[str] = []
    header = (
        "{:<12} {:>6} {:>9} {:>8} {:>8} {:>16} {:>18} {:>18} {:>10} {:>15} {:>16} {:>9} {:>11}".format(
            "difficulty",
            "count",
            "exec_acc",
            "correct",
            "wrong",
            "correct_refusal",
            "incorrect_refusal",
            "violation_correct",
            "violation_wrong",
            "sql_err",
            "testcase_fail",
            "timeout",
            "viol_rate",
        )
    )
    print(header)
    lines.append(header)

    for level in [*DIFFICULTY_LEVELS, "all"]:
        bucket = stats.get(level)
        if not bucket:
            continue
        exec_acc = compute_exec_accuracy(bucket)
        viol_rate = compute_violation_rate(bucket)
        row = "{:<12} {:>6} {:>9.4f} {:>8} {:>8} {:>16} {:>18} {:>18} {:>10} {:>15} {:>16} {:>9} {:>11.4f}".format(
            level,
            int(bucket["count"]),
            exec_acc,
            int(bucket["correct"]),
            int(bucket["wrong"]),
            int(bucket["correct_refusal"]),
            int(bucket["incorrect_refusal"]),
            int(bucket["violation_correct"]),
            int(bucket["violation_wrong"]),
            int(bucket["sql_error"]),
            int(bucket["testcase_failed"]),
            int(bucket["timeout"]),
            viol_rate,
        )
        print(row)
        lines.append(row)

    overall = stats.get("all", init_bucket())
    access_control = compute_access_control_metrics(overall)
    sql_metrics = compute_sql_metrics(overall)
    exec_accuracy = compute_exec_accuracy(overall)
    per_difficulty_metrics = {}
    for level in DIFFICULTY_LEVELS:
        bucket = stats.get(level)
        if bucket and bucket["count"]:
            per_difficulty_metrics[level] = {
                "bucket": bucket,
                "access": compute_access_control_metrics(bucket),
                "sql": compute_sql_metrics(bucket),
                "exec_accuracy": compute_exec_accuracy(bucket),
                "answerable": compute_answerable(bucket),
            }
    summary_lines = [
        "",
        "Overall summary:",
        f"  Total samples           : {int(overall['count'])}",
        f"  Answerable samples      : {int(compute_answerable(overall))}",
        f"  Execution accuracy      : {exec_accuracy:.4f}",
        f"  Correct refusals        : {int(overall['correct_refusal'])}",
        f"  Violations (any)        : {int(overall['violation_correct'] + overall['violation_wrong'])} ({access_control['violation_rate']:.4f})",
        f"  Missing predictions     : {int(overall['missing'])}",
        f"  Empty SQL submissions   : {int(overall['empty'])}",
        "  Access Control metrics:",
        f"    TP / FP / FN / TN      : {access_control['tp']} / {access_control['fp']} / {access_control['fn']} / {access_control['tn']}",
        f"    Precision / Recall / F1: {access_control['precision']:.4f} / {access_control['recall']:.4f} / {access_control['f1']:.4f}",
        f"    Accuracy               : {access_control['accuracy']:.4f}",
        f"    Violation rate         : {access_control['violation_rate']:.4f}",
        f"    Over-refusal rate      : {access_control['over_refusal_rate']:.4f}",
        "  SQL performance metrics:",
        f"    SQL-emitting samples   : {sql_metrics['sql_attempts']}",
        f"    Correct SQL (incl. VC) : {sql_metrics['correct_sql']}",
        f"    SQL accuracy (SQL)     : {sql_metrics['sql_accuracy']:.4f}",
        f"    Safe SQL accuracy      : {sql_metrics['safe_sql_accuracy']:.4f}",
        f"    SafeEX / UnsafeEX      : {sql_metrics['safe_ex']:.4f} / {sql_metrics['unsafe_ex']:.4f}",
    ]
    lines.extend(summary_lines)
    print()
    for line in summary_lines[1:]:
        print(line)

    if per_difficulty_metrics:
        per_diff_lines: List[str] = ["", "Per-difficulty access control & SQL metrics:"]
        for level in DIFFICULTY_LEVELS:
            metrics = per_difficulty_metrics.get(level)
            if not metrics:
                continue
            bucket = metrics["bucket"]
            access = metrics["access"]
            sql_detail = metrics["sql"]
            per_diff_lines.extend(
                [
                    f"  [{level}] count={int(bucket['count'])} answerable={int(metrics['answerable'])} exec_acc={metrics['exec_accuracy']:.4f}",
                    "    Six-category (C/W/CR/IR/VC/VW): {} / {} / {} / {} / {} / {}".format(
                        int(bucket["correct"]),
                        int(bucket["wrong"]),
                        int(bucket["correct_refusal"]),
                        int(bucket["incorrect_refusal"]),
                        int(bucket["violation_correct"]),
                        int(bucket["violation_wrong"]),
                    ),
                    "    Access Control -> TP/FP/FN/TN: {} / {} / {} / {} | Precision={:.4f} Recall={:.4f} F1={:.4f} Acc={:.4f} | ViolationRate={:.4f} OverRefusalRate={:.4f}".format(
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
                    ),
                    "    SQL performance: attempts={} correct={} SQLAcc={:.4f} SafeSQLAcc={:.4f} SafeEX={:.4f} UnsafeEX={:.4f}".format(
                        sql_detail["sql_attempts"],
                        sql_detail["correct_sql"],
                        sql_detail["sql_accuracy"],
                        sql_detail["safe_sql_accuracy"],
                        sql_detail["safe_ex"],
                        sql_detail["unsafe_ex"],
                    ),
                ]
            )
        lines.extend(per_diff_lines)
        for line in per_diff_lines[1:]:
            print(line)

    info_lines: List[str] = []
    if skipped_instances:
        info_lines = ["", f"Skipped instances not present in predictions: {len(skipped_instances)}"]
        preview = list(skipped_instances[:10])
        if preview:
            info_lines.append("Preview: " + ", ".join(preview))
        for line in info_lines:
            print(line)
    lines.extend(info_lines)

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _normalise_run_name(name: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip()]
    normalised = "".join(cleaned).strip("_")
    return normalised or "run"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LiveSQLBench predictions")
    parser.add_argument("--predictions", required=True, help="Path to model predictions (JSON/JSONL)")
    parser.add_argument(
        "--base-data",
        default="./data/livesqlbench-base-lite-sqlite/livesqlbench_data_sqlite.jsonl",
        help="Path to LiveSQLBench base data jsonl",
    )
    parser.add_argument(
        "--gold-sql",
        default="./data/livesqlbench-base-lite-sqlite/livesqlbench_sqlite_gt_kg_testcases_0528.jsonl",
        help="Path to LiveSQLBench gold SQL jsonl",
    )
    parser.add_argument(
        "--db-root",
        default="./data/livesqlbench-base-lite-sqlite",
        help="Root directory containing per-database SQLite templates",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-example timeout in seconds")
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate only the first N samples")
    parser.add_argument(
        "--output-dir",
        default="./dbgpt_hub_sql/output/eval_result",
        help="Directory where evaluation summaries will be written",
    )
    parser.add_argument(
        "--run-name",
        help="Override the base name used for summary/incorrect output files",
    )

    args = parser.parse_args()

    predictions_path = Path(args.predictions).expanduser()
    base_data_path = Path(args.base_data).expanduser()
    gold_sql_path = Path(args.gold_sql).expanduser()
    db_root = Path(args.db_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    _ensure_output_dir(output_dir)

    stats, incorrect_logs, skipped_instances = evaluate_predictions(
        predictions_path=predictions_path,
        base_data_path=base_data_path,
        gold_sql_path=gold_sql_path,
        db_root=db_root,
        timeout=args.timeout,
        max_samples=args.max_samples,
    )

    if args.run_name:
        base_name = _normalise_run_name(args.run_name)
    else:
        base_name = os.path.splitext(os.path.basename(predictions_path))[0]
    summary_path = output_dir / f"{base_name}_livesqlbench_evaluate_result.txt"
    incorrect_path = output_dir / f"{base_name}_livesqlbench_incorrect.txt"

    print_and_save_summary(stats, summary_path, skipped_instances=skipped_instances)

    if incorrect_logs:
        incorrect_path.write_text("\n".join(incorrect_logs), encoding="utf-8")
    else:
        incorrect_path.write_text("All predictions correct.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
