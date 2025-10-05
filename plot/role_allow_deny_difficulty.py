#!/usr/bin/env python3
"""Draw horizontal stacked bars for allow/deny split per dataset & difficulty."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
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

COLOR_ALLOW = "#1b9e77"  # Teal from ColorBrewer Set2
COLOR_DENY = "#d95f02"  # Burnt orange for deny responses
BAR_HEIGHT = 0.48
AXIS_FACE_COLOR = "#f6f6f6"
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


@lru_cache(maxsize=1)
def load_spider_question_difficulties() -> Dict[tuple[str, str], str]:
    lookup: Dict[tuple[str, str], str] = {}
    spider_root = ROOT / "data" / "spider"
    candidates = [
        spider_root / "dev.json",
        spider_root / "test.json",
        spider_root / "train_spider.json",
        spider_root / "train_others.json",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"Warning: failed to load Spider split {path}: {exc}")
            continue

        for entry in payload:
            db_id = (entry.get("db_id") or "").strip()
            question = (entry.get("question") or "").strip()
            sql_dict = entry.get("sql")
            if not db_id or not question or not isinstance(sql_dict, dict):
                continue
            try:
                difficulty = compute_sql_difficulty(sql_dict)
            except Exception as exc:  # pragma: no cover - defensive guard
                print(f"Warning: failed to infer Spider difficulty for {db_id}: {exc}")
                continue
            lookup[(db_id, question)] = str(difficulty).lower()

    return lookup


def load_dataset_counts(config: DatasetConfig) -> Dict[str, Dict[str, Counter]]:
    counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    schema_cache: Dict[str, Schema] = {}
    spider_lookup: Dict[tuple[str, str], str] | None = None

    if config.difficulty_strategy == "spider_sql":
        spider_lookup = load_spider_question_difficulties()

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

                if difficulty is None and spider_lookup is not None:
                    question = (item.get("input") or item.get("question") or "").strip()
                    difficulty = spider_lookup.get((db_id, question))

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

    dataset_rows: Dict[str, List[tuple]] = defaultdict(list)

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
            allow_cnt = counts["allow"]
            deny_cnt = counts["deny"]
            allow_pct = allow_cnt / total * 100.0
            deny_pct = deny_cnt / total * 100.0
            dataset_rows[cfg.name].append(
                (
                    diff.title(),
                    allow_cnt,
                    deny_cnt,
                    allow_pct,
                    deny_pct,
                    total,
                )
            )

    if not dataset_rows:
        raise RuntimeError("No data available for plotting.")

    dataset_order = [cfg.name for cfg in DATASETS if cfg.name in dataset_rows]
    if not dataset_order:
        raise RuntimeError("No datasets available for plotting.")

    height_units = [max(len(dataset_rows[name]), 1) for name in dataset_order]
    total_units = sum(height_units)
    fig_height = max(6.0, 1.2 + BAR_HEIGHT * 2.1 * total_units)
    fig, axes = plt.subplots(
        len(dataset_order),
        1,
        figsize=(8.2, fig_height),
        sharex=False,
        gridspec_kw={"height_ratios": height_units},
    )
    if len(dataset_order) == 1:
        axes = [axes]

    for ax, dataset_key in zip(axes, dataset_order):
        rows = dataset_rows[dataset_key]
        display_name = dataset_display[dataset_key]
        y_positions = np.arange(len(rows))
        allow_counts = [row[1] for row in rows]
        deny_counts = [row[2] for row in rows]
        allow_pct = [row[3] for row in rows]
        deny_pct = [row[4] for row in rows]
        totals = [row[5] for row in rows]
        labels = [row[0] for row in rows]

        ax.set_facecolor(AXIS_FACE_COLOR)

        ax.barh(
            y_positions,
            allow_counts,
            color=COLOR_ALLOW,
            edgecolor="white",
            label="Allow",
            height=BAR_HEIGHT,
        )
        ax.barh(
            y_positions,
            deny_counts,
            left=allow_counts,
            color=COLOR_DENY,
            edgecolor="white",
            label="Deny",
            height=BAR_HEIGHT,
        )

        max_total = max(totals) if totals else 0
        x_limit = max_total * 1.15 if max_total > 0 else 1
        ax.set_xlim(0, x_limit)
        if max_total > 0:
            step = max(1, int(np.ceil(max_total / 5)))
            ax.set_xticks(np.arange(0, x_limit, step))
        ax.set_xlabel("Role responses (count)")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_ylim(-0.5, len(rows) - 0.5)

        for y, a_cnt, d_cnt, a_pct, d_pct, total in zip(y_positions, allow_counts, deny_counts, allow_pct, deny_pct, totals):
            if a_cnt > 0:
                text = f"{a_pct:.0f}%\n({int(a_cnt)})"
                if a_cnt > total * 0.15:
                    ax.text(a_cnt / 2, y, text, ha="center", va="center", color="white", fontsize=8)
                else:
                    ax.text(a_cnt + max(total * 0.01, 0.5), y, text, ha="left", va="center", color=COLOR_ALLOW, fontsize=8)
            if d_cnt > 0:
                text = f"{d_pct:.0f}%\n({int(d_cnt)})"
                mid = a_cnt + d_cnt / 2
                if d_cnt > total * 0.15:
                    ax.text(mid, y, text, ha="center", va="center", color="white", fontsize=8)
                else:
                    ax.text(a_cnt + d_cnt + max(total * 0.01, 0.5), y, text, ha="left", va="center", color=COLOR_DENY, fontsize=8)
            ax.text(total + max(total * 0.02, 0.6), y, f"Total {int(total)}", ha="left", va="center", color="#333333", fontsize=8)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower right", fontsize=8)
        ax.set_title(f"{display_name}")

    fig.suptitle("Role allow/deny distribution by difficulty", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = ROOT / "plot" / "role_allow_deny_difficulty.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved stacked bar plot to {output_path}")


if __name__ == "__main__":
    main()
