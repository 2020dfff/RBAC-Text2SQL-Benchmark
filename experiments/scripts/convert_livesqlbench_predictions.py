#!/usr/bin/env python3
"""Convert LiveSQLBench predictions into JSONL format for evaluation.

This utility aligns line-delimited SQL predictions with the official
LiveSQLBench metadata so that the evaluation scripts can locate the
corresponding ``instance_id`` (and role information when available).

Typical usage:

.. code-block:: bash

    python dbgpt_hub_sql/scripts/convert_livesqlbench_predictions.py \
        --pred-sql dbgpt_hub_sql/output/pred/pred_deepseek-coder_livesqlbench.sql \
        --dataset dbgpt_hub_sql/data/livesqlbench/dev/livesqlbench_data_sqlite.jsonl \
        --output dbgpt_hub_sql/output/pred/pred_deepseek-coder_livesqlbench.jsonl

For role-based predictions, supply the role-augmented dataset JSON instead::

    python dbgpt_hub_sql/scripts/convert_livesqlbench_predictions.py \
    --pred-sql dbgpt_hub_sql/output/pred/pred_deepseek-coder_livesqlbench_role.sql \
        --dataset dbgpt_hub_sql/data/livesqlbench/livesqlbench_dev_with_role.json \
        --output dbgpt_hub_sql/output/pred/pred_deepseek-coder_livesqlbench_role.jsonl

The script performs consistency checks on the number of predictions and
injects helpful metadata (database id, role name, difficulty) when present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from experiments.predict.response_cleaner import clean_model_response


def load_dataset(path: Path) -> List[Mapping[str, Any]]:
    """Load a JSON/JSONL dataset into a list of dictionaries."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records: List[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
                if not isinstance(obj, Mapping):
                    raise ValueError(f"Line {line_no} of {path} is not a JSON object")
                records.append(obj)
        return records

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        raise ValueError(f"Unsupported dataset extension: {path.suffix}")

    if isinstance(payload, list):
        return [obj for obj in payload if isinstance(obj, Mapping)]

    if isinstance(payload, Mapping):
        # Flatten mapping of lists/dicts (e.g., {"train": [...]}).
        items: List[Mapping[str, Any]] = []
        for value in payload.values():
            if isinstance(value, list):
                items.extend(obj for obj in value if isinstance(obj, Mapping))
            elif isinstance(value, Mapping):
                items.append(value)
        if items:
            return items

    raise ValueError(f"Unsupported JSON dataset structure in {path}")


def _extract_nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def retrieve_instance_id(entry: Mapping[str, Any]) -> str:
    candidates: Iterable[Any] = (
        entry.get("instance_id"),
        _extract_nested(entry, "metadata", "instance_id"),
        _extract_nested(entry, "metadata", "metadata", "instance_id"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise KeyError("Dataset entry missing instance_id")


def build_instruction_mapping(entries: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in entries:
        instruction = entry.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        try:
            instance_id = retrieve_instance_id(entry)
        except KeyError:
            continue

        key = instruction.strip()
        existing = mapping.get(key)
        if existing is not None and existing != instance_id:
            raise ValueError(
                "Inconsistent instance_id for instruction: "
                f"{key!r} maps to both {existing!r} and {instance_id!r}"
            )
        mapping[key] = instance_id
    return mapping


def collect_metadata(entry: Mapping[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    for key in ("db_id", "selected_database"):
        value = entry.get(key) or _extract_nested(entry, "metadata", key)
        if isinstance(value, str) and value.strip():
            metadata["db_id"] = value.strip()
            break

    role = entry.get("role")
    if not role:
        role = _extract_nested(entry, "metadata", "role")
        if not role:
            role = _extract_nested(entry, "metadata", "role_name")
    if isinstance(role, str) and role.strip():
        metadata["role"] = role.strip()

    difficulty = (
        entry.get("difficulty")
        or entry.get("difficulty_tier")
        or _extract_nested(entry, "metadata", "difficulty")
        or _extract_nested(entry, "metadata", "difficulty_tier")
    )
    if isinstance(difficulty, str) and difficulty.strip():
        metadata["difficulty"] = difficulty.strip()

    return metadata


def align_predictions(
    predictions_path: Path,
    dataset_entries: Sequence[Mapping[str, Any]],
    instruction_to_id: Optional[Mapping[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")

    with predictions_path.open("r", encoding="utf-8") as handle:
        predictions = [line.rstrip("\n") for line in handle]

    if len(predictions) != len(dataset_entries):
        raise ValueError(
            "Prediction count does not match dataset length: "
            f"{len(predictions)} predictions vs {len(dataset_entries)} entries"
        )

    results: List[Dict[str, Any]] = []
    sanitized = 0
    emptied = 0
    for idx, (entry, prediction) in enumerate(zip(dataset_entries, predictions), start=1):
        try:
            instance_id = retrieve_instance_id(entry)
        except KeyError:
            instruction = entry.get("instruction")
            if (
                instruction_to_id is not None
                and isinstance(instruction, str)
                and instruction.strip() in instruction_to_id
            ):
                instance_id = instruction_to_id[instruction.strip()]
            else:
                raise
        raw_prediction = prediction.strip()
        normalized_prediction = clean_model_response(raw_prediction)

        if normalized_prediction:
            if normalized_prediction != raw_prediction:
                sanitized += 1
            cleaned_prediction = normalized_prediction
        else:
            if raw_prediction:
                emptied += 1
                cleaned_prediction = raw_prediction
            else:
                cleaned_prediction = ""

        payload: Dict[str, Any] = {
            "instance_id": instance_id,
            "prediction": cleaned_prediction,
            "prediction_text": cleaned_prediction,
        }
        metadata = collect_metadata(entry)
        if metadata:
            payload["metadata"] = metadata
        results.append(payload)

    stats = {"sanitized": sanitized, "empty_after_clean": emptied}

    return results, stats


def save_jsonl(records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LiveSQLBench predictions into JSONL format")
    parser.add_argument("--pred-sql", required=True, help="Path to raw or cleaned .sql prediction file")
    parser.add_argument("--dataset", required=True, help="Path to LiveSQLBench dataset (JSON/JSONL)")
    parser.add_argument("--output", required=True, help="Destination JSONL file for evaluation")
    parser.add_argument(
        "--role-dataset",
        help="Optional role-augmented dataset providing instance_id if the primary dataset lacks it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictions_path = Path(args.pred_sql).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    dataset_entries = load_dataset(dataset_path)

    instruction_to_id: Optional[Mapping[str, str]] = None
    if args.role_dataset:
        role_dataset_path = Path(args.role_dataset).expanduser().resolve()
        role_entries = load_dataset(role_dataset_path)
        instruction_to_id = build_instruction_mapping(role_entries)

    aligned_records, stats = align_predictions(
        predictions_path,
        dataset_entries,
        instruction_to_id=instruction_to_id,
    )
    save_jsonl(aligned_records, output_path)

    print(
        f"Wrote {len(aligned_records)} records to {output_path}",
        flush=True,
    )

    if stats["sanitized"] or stats["empty_after_clean"]:
        print(
            "Sanitized {sanitized} predictions; {emptied} became empty after cleaning.".format(
                sanitized=stats["sanitized"], emptied=stats["empty_after_clean"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
