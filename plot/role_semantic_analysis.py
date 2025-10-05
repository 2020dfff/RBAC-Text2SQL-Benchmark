#!/usr/bin/env python3
"""Visualize role coverage and semantic alignment across datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
ROOT = Path(__file__).resolve().parents[1]

from matplotlib.colors import PowerNorm
import numpy as np

try:  # Prefer package import when available
    from plot.embedding_utils import cosine_similarity, embed_texts
except ModuleNotFoundError:  # Fallback for direct script execution
    from embedding_utils import cosine_similarity, embed_texts

import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

EVAL_DIR = ROOT / "experiments" / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.append(str(EVAL_DIR))

from process_sql import Schema, get_schema, get_sql


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display_name: str
    schema_loader: Callable[[], Dict[str, Dict[str, object]]]
    assignment_dir: Path
    assignment_patterns: Tuple[str, ...]
    default_output: Path
    dbs_per_panel: int = 20
    role_sql_patterns: Tuple[str, ...] | None = None
    difficulty_strategy: str = "none"  # 'precomputed', 'spider_sql', or 'none'


MODEL_PRESETS = {
    "baai": ("fastembed", "BAAI/bge-small-en-v1.5"),
    "openai": ("openai", "text-embedding-3-small"),
}


WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)

AGG_OPS = ("none", "max", "min", "count", "sum", "avg")


def _has_agg(unit) -> bool:
    return isinstance(unit, (list, tuple)) and len(unit) > 0 and unit[0] != AGG_OPS.index("none")


def _count_agg(units) -> int:
    return len([unit for unit in units if _has_agg(unit)])


def _get_nested_sql(sql_dict) -> List[dict]:
    nested: List[dict] = []
    cond_units = sql_dict["from"]["conds"][::2] + sql_dict["where"][::2] + sql_dict["having"][::2]
    for cond_unit in cond_units:
        if len(cond_unit) >= 4 and isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
        if len(cond_unit) >= 5 and isinstance(cond_unit[4], dict):
            nested.append(cond_unit[4])
    for key in ("intersect", "except", "union"):
        if sql_dict.get(key) is not None:
            nested.append(sql_dict[key])
    return nested


def _count_component1(sql_dict) -> int:
    count = 0
    if len(sql_dict["where"]) > 0:
        count += 1
    if len(sql_dict["groupBy"]) > 0:
        count += 1
    if len(sql_dict["orderBy"]) > 0:
        count += 1
    if sql_dict["limit"] is not None:
        count += 1
    if len(sql_dict["from"]["table_units"]) > 0:
        count += len(sql_dict["from"]["table_units"]) - 1

    ao_tokens = sql_dict["from"]["conds"][1::2] + sql_dict["where"][1::2] + sql_dict["having"][1::2]
    count += len([token for token in ao_tokens if token == "or"])

    cond_units = sql_dict["from"]["conds"][::2] + sql_dict["where"][::2] + sql_dict["having"][::2]
    count += len(
        [unit for unit in cond_units if len(unit) > 1 and unit[1] == WHERE_OPS.index("like")]
    )
    return count


def _count_component2(sql_dict) -> int:
    return len(_get_nested_sql(sql_dict))


def _count_others(sql_dict) -> int:
    count = 0
    agg_count = _count_agg(sql_dict["select"][1])
    agg_count += _count_agg(sql_dict["where"][::2])
    agg_count += _count_agg(sql_dict["groupBy"])
    if len(sql_dict["orderBy"]) > 0:
        agg_count += _count_agg(
            [unit[1] for unit in sql_dict["orderBy"][1] if unit[1]]
            + [unit[2] for unit in sql_dict["orderBy"][1] if unit[2]]
        )
    agg_count += _count_agg(sql_dict["having"])
    if agg_count > 1:
        count += 1

    if len(sql_dict["select"][1]) > 1:
        count += 1
    if len(sql_dict["where"]) > 1:
        count += 1
    if len(sql_dict["groupBy"]) > 1:
        count += 1
    return count


def compute_sql_difficulty(sql_dict) -> str:
    comp1 = _count_component1(sql_dict)
    comp2 = _count_component2(sql_dict)
    others = _count_others(sql_dict)

    if comp1 <= 1 and others == 0 and comp2 == 0:
        return "easy"
    if ((others <= 2 and comp1 <= 1 and comp2 == 0) or (comp1 <= 2 and others < 2 and comp2 == 0)):
        return "medium"
    if (
        (others > 2 and comp1 <= 2 and comp2 == 0)
        or (2 < comp1 <= 3 and others <= 2 and comp2 == 0)
        or (comp1 <= 1 and others == 0 and comp2 <= 1)
    ):
        return "hard"
    return "extra"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_table_summaries_from_tables_json(tables_path: Path) -> Dict[str, Dict[str, object]]:
    with tables_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    summaries: Dict[str, Dict[str, object]] = {}
    for entry in payload:
        db_id = entry.get("db_id")
        if not isinstance(db_id, str):
            continue
        table_names = entry.get("table_names", [])
        column_names = [col for _, col in entry.get("column_names", []) if col != "*"]
        summary_text = " ".join(table_names + column_names)
        summaries[db_id] = {
            "table_count": len(table_names),
            "summary_text": summary_text,
            "summary_tokens": set(tokenize(summary_text)),
        }
    return summaries


def load_livesql_schema_summaries(base_dir: Path) -> Dict[str, Dict[str, object]]:
    summaries: Dict[str, Dict[str, object]] = {}
    for db_dir in sorted(base_dir.iterdir()):
        if not db_dir.is_dir():
            continue
        db_id = db_dir.name
        schema_files = sorted(db_dir.glob("*_schema.txt"))
        if not schema_files:
            continue
        schema_text = "\n".join(path.read_text(encoding="utf-8") for path in schema_files)
        table_count = len(re.findall(r"(?i)\bcreate\s+table\b", schema_text))
        summaries[db_id] = {
            "table_count": table_count,
            "summary_text": schema_text,
            "summary_tokens": set(tokenize(schema_text)),
        }
    return summaries


def ensure_schema_embeddings(
    schema_meta: MutableMapping[str, MutableMapping[str, object]],
    *,
    embed_model: str | None = None,
    embed_backend: str | None = None,
    openai_api_key: str | None = None,
) -> None:
    pending_texts: List[str] = []
    pending_keys: List[str] = []
    for db_id, meta in schema_meta.items():
        if "embedding" in meta:
            continue
        summary_text = str(meta.get("summary_text", "")).strip()
        if summary_text:
            pending_keys.append(db_id)
            pending_texts.append(summary_text)
        else:
            meta["embedding"] = None

    if not pending_texts:
        return

    for db_id, emb in zip(
        pending_keys,
        embed_texts(
            pending_texts,
            model_name=embed_model or os.environ.get("ROLE_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            backend=embed_backend,
            openai_api_key=openai_api_key,
        ),
    ):
        schema_meta[db_id]["embedding"] = emb


def build_difficulty_lookup(config: DatasetConfig) -> Dict[Tuple[str, str], Counter]:
    if not config.role_sql_patterns:
        return {}

    difficulty_map: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    schema_cache: Dict[str, Schema] = {}
    spider_db_dir = ROOT / "data/spider/database"

    for pattern in config.role_sql_patterns:
        for path in sorted(config.assignment_dir.glob(pattern)):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    entries = json.load(handle)
            except Exception as err:
                print(f"Warning: failed to load role SQL dataset {path}: {err}")
                continue

            for item in entries:
                role_name = (item.get("role") or "").strip()
                db_id = item.get("db_id") or ""
                if not role_name or not db_id:
                    continue

                difficulty = item.get("difficulty")

                if difficulty is None and config.difficulty_strategy == "spider_sql":
                    sql_text = (item.get("output") or "").strip()
                    if not sql_text or sql_text.lower().startswith("sorry"):
                        continue
                    try:
                        schema = schema_cache.get(db_id)
                        if schema is None:
                            db_path = spider_db_dir / db_id / f"{db_id}.sqlite"
                            schema = Schema(get_schema(str(db_path)))
                            schema_cache[db_id] = schema
                        parsed_sql = get_sql(schema, sql_text)
                        difficulty = compute_sql_difficulty(parsed_sql)
                    except Exception as err:
                        print(
                            f"Warning: difficulty inference failed for {db_id}/{role_name}: {err}. Skipping."
                        )
                        continue

                if difficulty is None:
                    continue

                difficulty_map[(db_id, role_name)][str(difficulty).lower()] += 1

    return difficulty_map


def load_role_assignments(assignments_dir: Path, patterns: Sequence[str]) -> Dict[str, List[Dict[str, str]]]:
    aggregated: MutableMapping[str, Dict[Tuple[str, Tuple[str, ...]], Dict[str, str]]] = defaultdict(dict)
    for pattern in patterns:
        for path in sorted(assignments_dir.glob(pattern)):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            assignments = payload.get("assignments", {})
            for db_id, roles in assignments.items():
                bucket = aggregated[db_id]
                for role_info in roles:
                    role_name = (role_info.get("role", "") or "").strip().lower()
                    table_tokens = tuple(
                        sorted({t.strip().lower() for t in role_info.get("tables", "").split(",") if t.strip()})
                    )
                    key = (role_name, table_tokens)
                    bucket.setdefault(key, role_info)
    return {db_id: list(bucket.values()) for db_id, bucket in aggregated.items()}


def prepare_role_records(
    assignments: Mapping[str, Sequence[Mapping[str, str]]],
    schema_meta: MutableMapping[str, MutableMapping[str, object]],
    *,
    embed_model: str | None = None,
    embed_backend: str | None = None,
    openai_api_key: str | None = None,
    difficulty_lookup: Mapping[Tuple[str, str], Counter] | None = None,
) -> List[Dict[str, object]]:
    ensure_schema_embeddings(
        schema_meta,
        embed_model=embed_model,
        embed_backend=embed_backend,
        openai_api_key=openai_api_key,
    )

    role_specs: List[Dict[str, object]] = []
    for db_id, role_list in assignments.items():
        meta = schema_meta.get(db_id, {"table_count": 0, "embedding": None})
        table_count = int(meta.get("table_count", 0) or 0)
        schema_embedding = meta.get("embedding")

        for role_info in role_list:
            role_name = role_info.get("role", "")
            description = role_info.get("description", "")
            raw_tables = role_info.get("tables", "")
            tables = [t.strip() for t in raw_tables.split(",") if t.strip()]
            role_text_parts = [role_name.strip(), description.strip()]
            role_text = " ".join(part for part in role_text_parts if part)
            if not role_text:
                role_text = role_name or description or db_id

            role_specs.append(
                {
                    "db_id": db_id,
                    "role": role_name,
                    "description": description,
                    "tables": tables,
                    "num_tables": len(tables),
                "db_table_count": table_count,
                "schema_embedding": schema_embedding,
                "role_text": role_text,
                "is_system_manager": role_name.strip().lower() == "systemmanager",
                "difficulty_counts": difficulty_lookup.get((db_id, role_name), Counter()) if difficulty_lookup else Counter(),
            }
        )

    role_embeddings = embed_texts(
        [spec["role_text"] for spec in role_specs],
        model_name=embed_model or os.environ.get("ROLE_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        backend=embed_backend,
        openai_api_key=openai_api_key,
    )

    records: List[Dict[str, object]] = []
    for spec, role_emb in zip(role_specs, role_embeddings):
        similarity_raw = cosine_similarity(role_emb, spec.get("schema_embedding"))
        similarity = max(0.0, min(1.0, (similarity_raw + 1.0) / 2.0))
        records.append(
            {
                "db_id": spec["db_id"],
                "role": spec["role"],
                "description": spec["description"],
                "tables": spec["tables"],
                "num_tables": spec["num_tables"],
                "db_table_count": spec["db_table_count"],
                "similarity": similarity,
                "is_system_manager": spec["is_system_manager"],
                "difficulty_counts": dict(spec["difficulty_counts"]),
                "difficulty_primary": spec["difficulty_counts"].most_common(1)[0][0]
                if spec["difficulty_counts"]
                else None,
            }
        )

    return records


def chunk_sequence(items: Sequence[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [list(items)]
    return [list(items[idx : idx + size]) for idx in range(0, len(items), size)]


def prepare_plot_data(
    records: Sequence[Mapping[str, object]],
    dbs_per_panel: int,
) -> Dict[str, object]:
    if not records:
        raise ValueError("No role records available for plotting.")

    db_to_entries: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    similarities: List[float] = []

    for rec in records:
        db_id = rec["db_id"]
        similarity = float(rec["similarity"])
        similarities.append(similarity)
        total_tables = float(rec.get("db_table_count") or 0)
        accessible_tables = float(rec.get("num_tables") or 0)
        ratio = accessible_tables / total_tables if total_tables else 0.0
        ratio = min(max(ratio, 0.0), 1.0)
        db_to_entries[db_id].append(
            {
                "similarity": similarity,
                "size": 90.0 + ratio * 220.0,
                "is_system_manager": bool(rec.get("is_system_manager", False)),
            }
        )

    if not db_to_entries:
        raise ValueError("No role records available for plotting.")

    db_ids = sorted(db_to_entries.keys())
    panels = chunk_sequence(db_ids, dbs_per_panel)
    panel_heights = [max(1.0, 0.35 * len(chunk) + 0.6) for chunk in panels]

    min_sim = min(similarities)
    max_sim = max(similarities)
    if math.isclose(min_sim, max_sim):
        max_sim = min_sim + 1e-6
    norm = PowerNorm(gamma=0.55, vmin=min_sim, vmax=max_sim)
    cmap = colormaps["viridis"]

    layout_height = sum(panel_heights) + 1.6

    return {
        "db_to_entries": db_to_entries,
        "panels": panels,
        "panel_heights": panel_heights,
        "min_sim": min_sim,
        "sim_span": max_sim - min_sim if not math.isclose(max_sim, min_sim) else 1.0,
        "norm": norm,
        "cmap": cmap,
        "layout_height": layout_height,
    }


def render_dataset(
    container: plt.Figure,
    dataset_name: str,
    data: Dict[str, object],
    *,
    show_xlabel: bool,
    title_y: float = 0.96,
    right_margin: float = 0.9,
) -> None:
    random.seed(13)
    panels: List[List[str]] = data["panels"]
    panel_heights: List[float] = data["panel_heights"]
    db_to_entries: Dict[str, List[Dict[str, object]]] = data["db_to_entries"]
    min_sim: float = data["min_sim"]
    sim_span: float = data["sim_span"] or 1.0
    norm = data["norm"]
    cmap = data["cmap"]

    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])

    gs = container.add_gridspec(len(panels), 1, height_ratios=panel_heights, hspace=0.18)
    axes: List[plt.Axes] = []

    max_label_chars = 1

    for panel_idx, db_chunk in enumerate(panels):
        ax = container.add_subplot(gs[panel_idx, 0])
        axes.append(ax)

        y_ticks: List[float] = []
        y_labels: List[str] = []

        for row_idx, db_id in enumerate(db_chunk):
            y = float(row_idx)
            y_ticks.append(y)
            y_labels.append(db_id)
            max_label_chars = max(max_label_chars, len(db_id))
            ax.axhline(y, color="#d0d0d0", linestyle="--", linewidth=0.9, zorder=0)

            for entry in db_to_entries[db_id]:
                similarity = entry["similarity"]
                base_x = (similarity - min_sim) / sim_span if sim_span else 0.5
                jittered_x = base_x + random.uniform(-0.012, 0.012)
                if entry.get("is_system_manager"):
                    jittered_y = y
                    ax.scatter(
                        jittered_x,
                        jittered_y,
                        s=48.0,
                        c="#d62728",
                        alpha=0.9,
                        edgecolors="#8c1c15",
                        linewidths=0.4,
                        marker="^",
                        zorder=3,
                    )
                else:
                    jittered_y = y + random.uniform(-0.16, 0.16)
                    ax.scatter(
                        jittered_x,
                        jittered_y,
                        s=entry["size"],
                        c=[similarity],
                        cmap=cmap,
                        norm=norm,
                        alpha=0.82,
                        edgecolors="#333333",
                        linewidths=0.35,
                        marker="o",
                    )

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.6, len(db_chunk) - 1 + 0.6)
        tick_positions = np.linspace(0.0, 1.0, 5)
        if sim_span:
            tick_labels = [f"{(min_sim + pos * sim_span):.2f}" for pos in tick_positions]
        else:
            tick_labels = [f"{min_sim:.2f}" for _ in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=11)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.45)
        ax.set_ylabel("Database", fontsize=12)

        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=11)

    for ax in axes:
        ax.label_outer()

    if show_xlabel:
        axes[-1].set_xlabel("Semantic similarity (rescaled to observed range)", fontsize=12)
    else:
        axes[-1].set_xlabel("")
        axes[-1].set_xticklabels([])

    top_margin = min(0.9, max(0.55, title_y - 0.04))
    container.suptitle(
        f"{dataset_name} roles: permission breadth vs. schema alignment",
        fontsize=15,
        y=title_y,
    )
    left_margin = min(0.4, max(0.22, 0.18 + max_label_chars * 0.006))
    container.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=0.12, hspace=0.08)

    cbar = container.colorbar(
        scalar_map,
        ax=axes,
        fraction=0.055,
        pad=0.015,
        location="right",
    )
    cbar.set_label("Role/schema similarity (cosine via BGE embeddings)", fontsize=12)
    cbar.ax.tick_params(labelleft=False, labelright=False, labelsize=10)


def plot_roles(
    records: Sequence[Mapping[str, object]],
    output_path: Path,
    dataset_name: str,
    dbs_per_panel: int,
) -> None:
    data = prepare_plot_data(records, dbs_per_panel)
    fig = plt.figure(figsize=(7.2, data["layout_height"]))
    render_dataset(fig, dataset_name, data, show_xlabel=True, title_y=0.96, right_margin=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roles_multi(
    dataset_payloads: Sequence[Tuple[str, Sequence[Mapping[str, object]], int]],
    output_path: Path,
) -> None:
    prepared: List[Tuple[str, Dict[str, object]]] = []
    height_contribs: List[float] = []

    for display_name, records, dbs_per_panel in dataset_payloads:
        data = prepare_plot_data(records, dbs_per_panel)
        prepared.append((display_name, data))
        height_contribs.append(data["layout_height"])

    max_height = max(height_contribs) if height_contribs else 6.0
    fig_width = sum(7.2 for _ in prepared)
    fig_height = max_height
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs_main = fig.add_gridspec(1, len(prepared), width_ratios=[1.0] * len(prepared), wspace=0.35)

    for idx, (display_name, data) in enumerate(prepared):
        subfig = fig.add_subfigure(gs_main[0, idx])
        render_dataset(subfig, display_name, data, show_xlabel=True, title_y=0.9, right_margin=0.88)

    fig.suptitle("Role semantic alignment overview", y=0.97, fontsize=17)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_dataset_configs() -> Dict[str, DatasetConfig]:
    spider_tables = ROOT / "data/spider/tables.json"
    bird_tables = ROOT / "data/Bird/dev_20240627/dev_tables.json"
    livesql_base = ROOT / "data/livesqlbench-base-lite-sqlite"

    return {
        "spider": DatasetConfig(
            name="spider",
            display_name="Spider",
            schema_loader=lambda: load_table_summaries_from_tables_json(spider_tables),
            assignment_dir=ROOT / "outputs/spider_data",
            assignment_patterns=("role_assignments_spider_dev_*.json",),
            default_output=ROOT / "plot/spider_roles_overview.png",
            dbs_per_panel=20,
            role_sql_patterns=("role_sql_dataset_spider_dev_*.json",),
            difficulty_strategy="spider_sql",
        ),
        "spider_test": DatasetConfig(
            name="spider_test",
            display_name="Spider Test",
            schema_loader=lambda: load_table_summaries_from_tables_json(ROOT / "data/spider/test_tables.json"),
            assignment_dir=ROOT / "outputs/spider_data",
            assignment_patterns=("role_assignments_spider_test_*.json",),
            default_output=ROOT / "plot/spider_test_roles_overview.png",
            dbs_per_panel=20,
            role_sql_patterns=("role_sql_dataset_spider_test_*.json",),
            difficulty_strategy="spider_sql",
        ),
        "bird": DatasetConfig(
            name="bird",
            display_name="Bird",
            schema_loader=lambda: load_table_summaries_from_tables_json(bird_tables),
            assignment_dir=ROOT / "outputs/bird_data",
            assignment_patterns=("role_assignments_bird_dev_*.json",),
            default_output=ROOT / "plot/bird_roles_overview.png",
            dbs_per_panel=15,
            role_sql_patterns=("role_sql_dataset_bird_*.json",),
            difficulty_strategy="precomputed",
        ),
        "livesqlbench": DatasetConfig(
            name="livesqlbench",
            display_name="LiveSQLBench",
            schema_loader=lambda: load_livesql_schema_summaries(livesql_base),
            assignment_dir=ROOT / "outputs/livesql_data",
            assignment_patterns=("role_assignments_*_livesql_dev.json",),
            default_output=ROOT / "plot/livesql_roles_overview.png",
            dbs_per_panel=18,
            role_sql_patterns=("role_sql_dataset_livesql_*.json",),
            difficulty_strategy="precomputed",
        ),
    }


def parse_args() -> argparse.Namespace:
    configs = build_dataset_configs()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        help="Dataset identifier or comma-separated list (e.g. spider or spider,bird)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output image. Defaults to dataset-specific location.",
    )
    parser.add_argument(
        "--embed-backend",
        choices=["fastembed", "openai"],
        help="Embedding backend override (defaults to env ROLE_EMBED_BACKEND or auto-detect)",
    )
    parser.add_argument(
        "--embed-model",
        help="Embedding model name override (defaults to env ROLE_EMBED_MODEL)",
    )
    parser.add_argument(
        "--embed-openai-key",
        help="Explicit OpenAI API key for embeddings (otherwise reads OPENAI_API_KEY env)",
    )
    parser.add_argument(
        "--model",
        choices=["baai", "openai"],
        help="Convenience preset: 'baai' uses FastEmbed BGE, 'openai' uses text-embedding-3-small",
    )
    args = parser.parse_args()
    args.configs = configs
    return args


def main() -> None:
    args = parse_args()
    configs: Dict[str, DatasetConfig] = args.configs

    dataset_keys = [item.strip() for item in args.dataset.split(",") if item.strip()]
    if not dataset_keys:
        raise SystemExit("No dataset specified.")

    unknown = [key for key in dataset_keys if key not in configs]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}")

    outputs: List[Tuple[str, Sequence[Mapping[str, object]], int]] = []
    embed_backend = args.embed_backend
    embed_model = args.embed_model
    embed_api_key = args.embed_openai_key

    if args.model:
        preset_backend, preset_model = MODEL_PRESETS[args.model]
        if embed_backend is None:
            embed_backend = preset_backend
        if embed_model is None:
            embed_model = preset_model

    for key in dataset_keys:
        config = configs[key]
        schema_meta = config.schema_loader()
        assignments = load_role_assignments(config.assignment_dir, config.assignment_patterns)
        difficulty_lookup = build_difficulty_lookup(config)
        records = prepare_role_records(
            assignments,
            schema_meta,
            embed_model=embed_model,
            embed_backend=embed_backend,
            openai_api_key=embed_api_key,
            difficulty_lookup=difficulty_lookup,
        )
        outputs.append((config.display_name, records, config.dbs_per_panel))

    if len(outputs) == 1:
        display_name, records, dbs_per_panel = outputs[0]
        output_path = args.output or configs[dataset_keys[0]].default_output
        plot_roles(records, output_path, display_name, dbs_per_panel)
        print(f"Saved {display_name} role plot to {output_path} (records: {len(records)})")
    else:
        joined = "_".join(dataset_keys)
        default_path = ROOT / f"plot/roles_overview_{joined}.png"
        output_path = args.output or default_path
        plot_roles_multi(outputs, output_path)
        total_records = sum(len(records) for _, records, _ in outputs)
        print(
            f"Saved combined role plot for {', '.join(dataset_keys)} to {output_path} (records: {total_records})"
        )


if __name__ == "__main__":
    main()
