#!/usr/bin/env python3
"""Draw horizontal stacked bars for allow/deny split per dataset & difficulty."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

EVAL_DIR = ROOT / "experiments" / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.append(str(EVAL_DIR))

from process_sql import Schema, get_schema, get_sql  # type: ignore  # noqa: E402

try:  # Prefer packaged import when available
    from plot.role_semantic_analysis import compute_sql_difficulty
except ModuleNotFoundError:  # Fallback when executed as a script
    from role_semantic_analysis import compute_sql_difficulty

COLOR_ALLOW = "#1f77b4"
COLOR_DENY = "#ff7f0e"
ORDER_DIFF = ["easy", "medium", "hard", "extra"]

DENY_PATTERNS = (
    "sorry",
    "cannot answer",
    "can't answer",
    "do not have access",
    "don't have access",
    "no access",
    "not able to answer",
    "i do not know",
    "i don't know",
    "cannot comply",
    "i can't help",
    "i am unable",
)


def classify_decision(text: str) -> str:
    norm = (text or "").strip().lower()
    if not norm:
        return "deny"
    for pattern in DENY_PATTERNS:
        if pattern in norm:
            return "deny"
    return "allow"


class DatasetConfig:
    def __init__(
        self,
        name: str,
        display: str,
        assignment_dir: Path,
        role_sql_patterns: Sequence[str],
        difficulty_strategy: str,
    ) -> None:
        self.name = name
        self.display = display
        self.assignment_dir = assignment_dir
        self.role_sql_patterns = role_sql_patterns
        self.difficulty_strategy = difficulty_strategy


DATASETS: List[DatasetConfig] = [
    DatasetConfig(
        "spider",
        "Spider",
        ROOT / "outputs" / "spider_data",
        ("role_sql_dataset_spider_dev_*.json",),
        "spider_sql",
    ),
    DatasetConfig(
        "bird",
        "Bird",
        ROOT / "outputs" / "bird_data",
        ("role_sql_dataset_bird_*.json",),
        "precomputed",
    ),
    DatasetConfig(
        "livesqlbench",
        "LiveSQLBench",
        ROOT / "outputs" / "livesql_data",
        ("role_sql_dataset_livesql_*.json",),
        "precomputed",
    ),
]


def infer_difficulty_spider(db_id: str, sql_text: str, schema_cache: Dict[str, Schema]) -> str | None:
    sql_text = (sql_text or "").strip()
    if not sql_text or sql_text.lower().startswith("sorry"):
        return None
    db_path = ROOT / "data" / "spider" / "database" / db_id / f"{db_id}.sqlite"
    try:
        schema = schema_cache.get(db_id)
        if schema is None:
            schema = Schema(get_schema(str(db_path)))
            schema_cache[db_id] = schema
        parsed = get_sql(schema, sql_text)
        return compute_sql_difficulty(parsed)
    except Exception as exc:
        print(f"Warning: failed to infer difficulty for {db_id}: {exc}")
        return None


def load_dataset_counts(config: DatasetConfig) -> Dict[str, Dict[str, Counter]]:
    counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    schema_cache: Dict[str, Schema] = {}

    for pattern in config.role_sql_patterns:
        for path in sorted(config.assignment_dir.glob(pattern)):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    entries = json.load(handle)
            except Exception as err:
                print(f"Warning: failed to load {path}: {err}")
                continue

            for item in entries:
                db_id = item.get("db_id") or ""
                role_name = (item.get("role") or "").strip()
                if not db_id or not role_name:
                    continue

                decision = classify_decision(item.get("output") or "")

                difficulty = item.get("difficulty")
                if difficulty is None and config.difficulty_strategy == "spider_sql":
                    difficulty = infer_difficulty_spider(db_id, item.get("output", ""), schema_cache)
                if difficulty is None:
                    difficulty = "unknown" if decision == "deny" else None
                if difficulty is None:
                    continue

                counts[db_id][str(difficulty).lower()][decision] += 1

    return counts


def sort_difficulties(difficulties: Iterable[str]) -> List[str]:
    seen = []
    for diff in ORDER_DIFF + sorted(set(difficulties)):
        if diff in difficulties and diff not in seen:
            seen.append(diff)
    return seen


def main() -> None:
    aggregated: Dict[str, Dict[str, Counter]] = {}
    dataset_display = {cfg.name: cfg.display for cfg in DATASETS}

    for cfg in DATASETS:
        aggregated[cfg.name] = load_dataset_counts(cfg)

    bars: List[tuple] = []  # dataset, difficulty, allow%, deny%, allow_cnt, deny_cnt

    for cfg in DATASETS:
        per_db = aggregated[cfg.name]
        difficulty_totals: Dict[str, Counter] = defaultdict(Counter)
        for diff_map in per_db.values():
            for diff, decisions in diff_map.items():
                difficulty_totals[diff].update(decisions)

        ordered = sort_difficulties(difficulty_totals.keys())
        for diff in ordered:
            counts = difficulty_totals[diff]
            total = counts["allow"] + counts["deny"]
            if total == 0:
                continue
            allow_pct = counts["allow"] / total * 100.0
            deny_pct = counts["deny"] / total * 100.0
            bars.append(
                (
                    dataset_display[cfg.name],
                    diff.title(),
                    allow_pct,
                    deny_pct,
                    counts["allow"],
                    counts["deny"],
                )
            )

    if not bars:
        raise RuntimeError("No data available for plotting.")

    # Prepare vertical positions with gaps between datasets
    y_positions = []
    y_labels = []
    current_y = 0.0
    last_dataset = None

    for dataset, difficulty, allow_pct, deny_pct, _, _ in bars:
        if last_dataset is not None and dataset != last_dataset:
            current_y += 0.4
        y_positions.append(current_y)
        y_labels.append(f"{dataset} · {difficulty}")
        current_y += 1.0
        last_dataset = dataset

    fig_height = 0.6 * len(y_positions) + 1.5
    fig, ax = plt.subplots(figsize=(10, fig_height))

    allow_values = [row[2] for row in bars]
    deny_values = [row[3] for row in bars]

    ax.barh(y_positions, allow_values, color=COLOR_ALLOW, edgecolor="white", label="Allow")
    ax.barh(y_positions, deny_values, left=allow_values, color=COLOR_DENY, edgecolor="white", label="Deny")

    for idx, (y, allow_pct, deny_pct) in enumerate(zip(y_positions, allow_values, deny_values)):
        if allow_pct > 3:
            ax.text(allow_pct / 2, y, f"{allow_pct:.0f}%", ha="center", va="center", color="white", fontsize=9)
        elif allow_pct > 0:
            ax.text(allow_pct + 1, y, f"{allow_pct:.0f}%", ha="left", va="center", color=COLOR_ALLOW, fontsize=8)
        if deny_pct > 3:
            ax.text(allow_pct + deny_pct / 2, y, f"{deny_pct:.0f}%", ha="center", va="center", color="white", fontsize=9)
        elif deny_pct > 0:
            ax.text(allow_pct + deny_pct - 1, y, f"{deny_pct:.0f}%", ha="right", va="center", color=COLOR_DENY, fontsize=8)

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Role responses (%)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right")

    ax.set_title("Role allow/deny distribution by difficulty")

    output_path = ROOT / "plot" / "role_allow_deny_difficulty.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved stacked bar plot to {output_path}")


if __name__ == "__main__":
    main()
