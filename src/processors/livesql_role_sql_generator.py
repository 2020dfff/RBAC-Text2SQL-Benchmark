"""Generate RBAC-aware Text2SQL dataset entries for LiveSQLBench."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .livesql_describer import LiveSQLDatabaseDescriber, LiveSQLDatabaseInfo
from .spider_role_sql_generator import RoleSQLGenerator

logger = logging.getLogger("livesql_role_sql")


class LiveSQLRoleSQLGenerator(RoleSQLGenerator):
    """Specialised RoleSQL generator for the LiveSQLBench dataset."""

    _CTE_NAME_PATTERN = re.compile(
        r"(?:WITH\s+(?:RECURSIVE\s+)?|,\s*)([A-Za-z_][\w]*)\s*(?:\([^)]*\))?\s+AS\s*\(",
        re.IGNORECASE | re.DOTALL,
    )
    _INLINE_SUBQUERY_ALIAS_PATTERN = re.compile(
        r"(?:FROM|JOIN)\s*\(\s*SELECT[\s\S]+?\)\s*(?:AS\s+)?([A-Za-z_][\w]*)",
        re.IGNORECASE,
    )
    _CREATE_OBJECT_PATTERN = re.compile(
        r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP|TEMPORARY\s+)?(?:TABLE|VIEW|INDEX|TRIGGER)\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w\.]*)",
        re.IGNORECASE,
    )
    _DROP_OBJECT_PATTERN = re.compile(
        r"\bDROP\s+(?:TABLE|VIEW|INDEX|TRIGGER)\s+(?:IF\s+EXISTS\s+)?([A-Za-z_][\w\.]*)",
        re.IGNORECASE,
    )
    _SELECT_PATTERN = re.compile(r"\bSELECT\b", re.IGNORECASE)
    _WRITE_PATTERN = re.compile(r"\b(INSERT|UPDATE|DELETE|REPLACE|UPSERT|MERGE)\b", re.IGNORECASE)
    _DDL_PATTERN = re.compile(r"\b(CREATE|DROP|ALTER|TRUNCATE|RENAME)\b", re.IGNORECASE)
    _TRANSACTION_PATTERN = re.compile(r"\b(BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE)\b", re.IGNORECASE)
    _UTILITY_PATTERN = re.compile(r"\b(ANALYZE|VACUUM|EXPLAIN|PRAGMA|ATTACH|DETACH)\b", re.IGNORECASE)
    FULL_ACCESS_ROLES = {"systemmanager"}

    def __init__(self, project_root: Path, describer: LiveSQLDatabaseDescriber, output_dir: Optional[str] = None):
        super().__init__(project_root, output_dir)
        self.describer = describer
        self.livesql_root = project_root / "data" / "livesqlbench-base-lite-sqlite"
        self.base_data_file = self.livesql_root / "livesqlbench_data_sqlite.jsonl"
        self.gt_data_file = self.livesql_root / "livesqlbench_sqlite_gt_kg_testcases_0528.jsonl"
        self._statement_type_counts: Counter[str] = Counter()
        self._last_total_examples: int = 0
        self._last_filtered_examples: int = 0

        self._schema_text_cache: Dict[str, str] = {}
        self._column_meanings_cache: Dict[str, Dict[str, object]] = {}
        self._knowledge_cache: Dict[str, Sequence[Dict[str, object]]] = {}
        self._prompt_base_cache: Dict[str, str] = {}
        self._compact_instruction_cache: Dict[str, str] = {}

        # Override data sources inherited from Spider settings
        self.data_sources = {"dev": self.base_data_file}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_livesql_role_sql_dataset(
        self,
        role_file_path: Optional[str] = None,
        data_source: str = "dev",
        allowed_statement_types: Optional[Set[str]] = None,
    ) -> List[Dict[str, object]]:
        """Combine LiveSQL examples with generated role assignments."""
        role_assignments = self.load_role_assignments(role_file_path)
        livesql_examples = self._load_livesql_examples(data_source)
        gt_map = self._load_ground_truth_map()

        normalized_allowed: Optional[Set[str]] = None
        if allowed_statement_types is not None:
            normalized_allowed = {item.lower() for item in allowed_statement_types}

        if not role_assignments:
            logger.error("No role assignments available; aborting dataset generation")
            return []

        dataset: List[Dict[str, object]] = []
        processed_dbs: Set[str] = set()
        skipped_dbs: Set[str] = set()
        missing_gold_sql = 0
        total_examples = 0
        filtered_examples = 0

        self._statement_type_counts = Counter()
        self._last_total_examples = 0
        self._last_filtered_examples = 0

        allowed_count = 0
        denied_count = 0

        for example in livesql_examples:
            db_id = example.get("selected_database")
            instance_id = example.get("instance_id")
            if not db_id or not instance_id:
                continue

            if db_id not in role_assignments:
                skipped_dbs.add(db_id)
                continue

            processed_dbs.add(db_id)
            instruction = self._build_instruction(example, db_id)

            gold = gt_map.get(instance_id, {})
            sql_statements = gold.get("sol_sql") or example.get("sol_sql") or []
            if not sql_statements:
                missing_gold_sql += 1
                continue

            sql_text = "\n".join(sql_statements)
            total_examples += 1

            statement_label, _ = self._classify_sql_text(sql_text)
            self._statement_type_counts[statement_label] += 1

            if normalized_allowed is not None and statement_label not in normalized_allowed:
                filtered_examples += 1
                continue

            query_tables, derived_refs, raw_tables = self._extract_query_tables(sql_text)

            raw_difficulty = (
                example.get("difficulty")
                or example.get("difficulty_tier")
                or gold.get("difficulty")
            )
            if isinstance(raw_difficulty, str):
                difficulty_value = raw_difficulty.strip().lower() or "unknown"
            elif raw_difficulty is None:
                difficulty_value = "unknown"
            else:
                difficulty_value = str(raw_difficulty)

            for role_info in role_assignments[db_id]:
                role_name = role_info.get("role")
                role_tables = role_info.get("tables", "")
                if not role_name:
                    continue

                ext_knowledge = gold.get("external_knowledge")
                if ext_knowledge is None:
                    ext_knowledge = example.get("external_knowledge")

                test_cases = gold.get("test_cases")
                if test_cases is None:
                    test_cases = example.get("test_cases")

                metadata: Dict[str, object] = {
                    "instance_id": instance_id,
                    "gold_sql": sql_text,
                    "raw_query_tables": sorted(raw_tables),
                    "query_tables": sorted(query_tables),
                    "difficulty": difficulty_value,
                }

                category = example.get("category")
                if category is not None:
                    metadata["category"] = category

                difficulty_tier = example.get("difficulty_tier")
                if difficulty_tier is not None:
                    metadata["difficulty_tier"] = difficulty_tier

                role_description = role_info.get("description")
                if role_description:
                    metadata["role_description"] = role_description

                high_level = example.get("high_level")
                if high_level is not None:
                    metadata["high_level"] = high_level

                for key in ("preprocess_sql", "clean_up_sqls"):
                    seq = example.get(key, [])
                    if seq:
                        metadata[key] = seq

                if ext_knowledge:
                    metadata["external_knowledge"] = ext_knowledge

                if test_cases:
                    metadata["test_cases"] = test_cases

                entry = {
                    "db_id": db_id,
                    "instruction": instruction,
                    "role": role_name,
                    "tables": role_tables,
                    "input": example.get("query", ""),
                    "output": "",
                    "difficulty": difficulty_value,
                    "metadata": metadata,
                }

                allowed, missing_tables = self._has_permission(
                    role_name,
                    query_tables,
                    role_tables,
                    derived_refs,
                    raw_tables,
                )

                if allowed:
                    entry["output"] = sql_text
                    allowed_count += 1
                else:
                    entry["output"] = "Sorry, I cannot answer."
                    denied_count += 1
                    if logger.isEnabledFor(logging.DEBUG) and missing_tables:
                        logger.debug(
                            "Permission denied for role %s on %s: missing tables %s",
                            role_name,
                            db_id,
                            sorted(missing_tables),
                        )

                dataset.append(entry)

        logger.info("LiveSQL dataset generation summary:")
        logger.info("- Databases processed: %d", len(processed_dbs))
        logger.info("- Databases skipped (no roles): %d", len(skipped_dbs))
        logger.info("- Examples produced: %d", len(dataset))
        logger.info("- Allowed queries: %d", allowed_count)
        logger.info("- Denied queries: %d", denied_count)
        if missing_gold_sql:
            logger.warning("- Examples skipped due to missing gold SQL: %d", missing_gold_sql)
        if skipped_dbs:
            logger.debug("Skipped databases: %s", sorted(skipped_dbs))

        self._last_total_examples = total_examples
        self._last_filtered_examples = filtered_examples

        return dataset

    def save_livesql_dataset(
        self,
        dataset: List[Dict[str, object]],
        output_path: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_path is None:
            output_path = str(self.output_dir / f"role_sql_dataset_livesql_{timestamp}_dev.json")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        logger.info("LiveSQL dataset saved to %s", out_path)
        logger.info("File size: %s bytes", out_path.stat().st_size)
        return str(out_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_query_tables(self, sql_text: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """Extract physical tables referenced by a SQL query while ignoring derived references."""

        raw_tables = super().extract_tables_from_query(sql_text)
        cte_names = self._extract_cte_names(sql_text)
        inline_aliases = self._extract_inline_aliases(sql_text)
        defined_objects = self._extract_defined_objects(sql_text)
        derived_refs = cte_names | inline_aliases | defined_objects
        filtered_tables = {
            table
            for table in raw_tables
            if table not in derived_refs
        }
        removed_tables = raw_tables - filtered_tables

        if logger.isEnabledFor(logging.DEBUG) and removed_tables:
            logger.debug(
                "Filtered derived tables from query: %s", sorted(removed_tables)
            )
        return filtered_tables, derived_refs, raw_tables

    def _extract_cte_names(self, sql_text: str) -> Set[str]:
        if not sql_text:
            return set()

        return {
            match.group(1).lower()
            for match in self._CTE_NAME_PATTERN.finditer(sql_text)
        }

    def _extract_inline_aliases(self, sql_text: str) -> Set[str]:
        if not sql_text:
            return set()

        return {
            match.group(1).lower()
            for match in self._INLINE_SUBQUERY_ALIAS_PATTERN.finditer(sql_text)
        }

    def _extract_defined_objects(self, sql_text: str) -> Set[str]:
        if not sql_text:
            return set()

        names = set()
        for pattern in (self._CREATE_OBJECT_PATTERN, self._DROP_OBJECT_PATTERN):
            for match in pattern.finditer(sql_text):
                normalized = self._normalize_identifier(match.group(1))
                if normalized:
                    names.add(normalized)
        return names

    @staticmethod
    def _normalize_identifier(name: str) -> str:
        if not name:
            return ""
        cleaned = name.strip().strip('"`[]')
        if not cleaned:
            return ""
        if "." in cleaned:
            cleaned = cleaned.split(".")[-1]
        return cleaned.lower()

    def _classify_sql_text(self, sql_text: str) -> Tuple[str, Set[str]]:
        if not sql_text:
            return "unknown", {"unknown"}

        categories: Set[str] = set()
        if self._SELECT_PATTERN.search(sql_text):
            categories.add("select")
        if self._WRITE_PATTERN.search(sql_text):
            categories.add("write")
        if self._DDL_PATTERN.search(sql_text):
            categories.add("ddl")
        if self._TRANSACTION_PATTERN.search(sql_text):
            categories.add("transaction")
        if self._UTILITY_PATTERN.search(sql_text):
            categories.add("utility")

        if not categories:
            categories.add("unknown")

        label = self._derive_statement_label(categories)
        return label, categories

    @staticmethod
    def _derive_statement_label(categories: Set[str]) -> str:
        if not categories:
            return "unknown"
        for key in ("write", "ddl", "transaction", "utility", "select", "unknown"):
            if key in categories:
                return key
        return sorted(categories)[0]

    def get_last_statement_type_counts(self) -> Dict[str, int]:
        return dict(self._statement_type_counts)

    def get_last_total_example_count(self) -> int:
        return self._last_total_examples

    def get_last_filtered_example_count(self) -> int:
        return self._last_filtered_examples

    def _has_permission(
        self,
        role_name: Optional[str],
        query_tables: Set[str],
        role_tables: str,
        derived_refs: Set[str],
        raw_tables: Set[str],
    ) -> Tuple[bool, Set[str]]:
        role_tables_set = {t.strip().lower() for t in role_tables.split(",") if t.strip()}
        missing_tables = {table for table in query_tables if table not in role_tables_set}

        if not missing_tables:
            return True, set()

        normalized_role = (role_name or "").strip().lower()
        if normalized_role in self.FULL_ACCESS_ROLES:
            raw_missing = {table for table in raw_tables if table not in role_tables_set}
            unresolved = {table for table in raw_missing if table not in derived_refs}
            if not unresolved:
                logger.debug(
                    "Granting full-access role %s after excluding derived references %s",
                    role_name,
                    sorted(raw_missing),
                )
                return True, set()
            missing_tables = unresolved

        return False, missing_tables

    def _load_livesql_examples(self, data_source: str) -> List[Dict[str, object]]:
        if data_source not in self.data_sources:
            raise ValueError(f"Unsupported LiveSQL data source: {data_source}")

        data_file = self.data_sources[data_source]
        if not data_file.exists():
            logger.error("LiveSQL data file not found: %s", data_file)
            return []

        examples: List[Dict[str, object]] = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line in %s: %s", data_file, exc)
        logger.info("Loaded %d LiveSQL examples from %s", len(examples), data_file)
        return examples

    def _load_ground_truth_map(self) -> Dict[str, Dict[str, object]]:
        gt_map: Dict[str, Dict[str, object]] = {}
        if not self.gt_data_file.exists():
            logger.warning("Ground-truth file not found for LiveSQL: %s", self.gt_data_file)
            return gt_map

        with open(self.gt_data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line in %s: %s", self.gt_data_file, exc)
                    continue
                instance_id = record.get("instance_id")
                if instance_id:
                    gt_map[instance_id] = record
        logger.info("Loaded ground-truth metadata for %d LiveSQL examples", len(gt_map))
        return gt_map

    def _build_instruction(self, example: Dict[str, object], db_id: str) -> str:
        query = example.get("query", "") or ""

        schema_text = self._load_schema_text(db_id)
        column_meanings = self._load_column_meanings(db_id)
        knowledge_entries = self._load_external_knowledge(db_id)

        column_meanings_str = json.dumps(column_meanings, indent=2, ensure_ascii=False)

        visible_kbs: List[Dict[str, object]] = []
        for entry in knowledge_entries:
            if not isinstance(entry, dict):
                continue
            visible = {field: entry[field] for field in ("id", "knowledge", "description", "definition") if field in entry}
            if visible:
                visible_kbs.append(visible)

        knowledge_str = json.dumps(visible_kbs, indent=2, ensure_ascii=False)

        return (
            "# Database Schema:\n"
            f"{schema_text}\n\n"
            "# Column Meanings:\n"
            f"{column_meanings_str}\n\n"
            "# External Knowledge:\n"
            f"{knowledge_str}\n\n"
            "# User Task:\n"
            f"{query}\n\n"
            "Generate the correct PostgreSQL to handle the user task above:\n"
            "(FORMAT: You should enclose your final PostgreSQL in '```postgresql\n[Your Generated SQLs]\n```' in the end. Could use semicolon to separate multiple statements.)\n\n"
            "# Your Generated SQL:\n"
            "```postgresql"
        )

    def _get_compact_instruction(self, db_id: str) -> str:
        if db_id in self._compact_instruction_cache:
            return self._compact_instruction_cache[db_id]

        prompt_base = self._get_prompt_base(db_id)

        compact_instruction = (
            f"{prompt_base}\n\n"
            "Follow these rules when responding to a user request:\n"
            "1. Use only PostgreSQL syntax.\n"
            "2. Respect the role's table permissions; avoid tables not listed for the role.\n"
            "3. Return only SQL wrapped inside a single ```postgresql``` block without extra commentary."
        )

        self._compact_instruction_cache[db_id] = compact_instruction
        return compact_instruction

    def _get_prompt_base(self, db_id: str) -> str:
        if db_id in self._prompt_base_cache:
            return self._prompt_base_cache[db_id]

        db_info = self.describer.load_database(db_id)
        schema_summary = self._summarize_schema_tables(db_info)
        column_summary = self._summarize_column_meanings(db_id, db_info)
        knowledge_summary = self._summarize_knowledge(db_id)

        prompt_base = (
            f"Database `{db_id}` reference:\n"
            f"{schema_summary}\n\n"
            "Column hints:\n"
            f"{column_summary}\n\n"
            "External knowledge notes:\n"
            f"{knowledge_summary}"
        )

        self._prompt_base_cache[db_id] = prompt_base
        return prompt_base

    def _summarize_schema_tables(self, db_info: Optional[LiveSQLDatabaseInfo]) -> str:
        if not db_info or not getattr(db_info, "tables", None):
            return "Schema summary unavailable."

        lines: List[str] = []
        for table in db_info.tables:
            column_names = [column.name for column in table.columns]
            preview = ", ".join(column_names[:6])
            if len(column_names) > 6:
                preview += ", ..."
            lines.append(f"- {table.name} ({len(column_names)} cols): {preview}")

        return "\n".join(lines) if lines else "Schema summary unavailable."

    def _summarize_column_meanings(
        self,
        db_id: str,
        db_info: Optional[LiveSQLDatabaseDescriber],
        max_tables: int = 6,
        max_columns: int = 4,
    ) -> str:
        meanings = self._load_column_meanings(db_id)
        if not meanings:
            return "No column meaning annotations available."

        table_name_map: Dict[str, str] = {}
        if db_info and getattr(db_info, "tables", None):
            for table in db_info.tables:
                table_name_map[table.name.lower()] = table.name

        grouped: defaultdict[str, List[Tuple[str, str]]] = defaultdict(list)
        for key, description in meanings.items():
            parts = key.split("|")
            if len(parts) >= 3:
                table_key = parts[-2].lower()
                column_key = parts[-1]
            elif len(parts) == 2:
                table_key = parts[0].lower()
                column_key = parts[1]
            else:
                continue

            table_label = table_name_map.get(table_key, parts[-2] if len(parts) >= 2 else table_key)

            if isinstance(description, dict):
                description_text = json.dumps(description, ensure_ascii=False)
            else:
                description_text = str(description)

            grouped[table_label].append((column_key, description_text))

        if not grouped:
            return "No column meaning annotations available."

        lines: List[str] = []
        for idx, (table_label, entries) in enumerate(sorted(grouped.items())):
            if idx >= max_tables:
                lines.append("...")
                break

            entries_sorted = sorted(entries, key=lambda item: item[0])[:max_columns]
            entry_text = "; ".join(
                f"{column}: {self._truncate_text(text, 80)}" for column, text in entries_sorted
            )
            lines.append(f"- {table_label}: {entry_text}")

        return "\n".join(lines) if lines else "No column meaning annotations available."

    def _summarize_knowledge(self, db_id: str, limit: int = 4) -> str:
        entries = self._load_external_knowledge(db_id)
        if not entries:
            return "No external knowledge provided."

        lines: List[str] = []
        for idx, entry in enumerate(entries):
            if idx >= limit:
                remaining = len(entries) - limit
                if remaining > 0:
                    lines.append(f"... (+{remaining} more)")
                break

            primary = entry.get("knowledge") or entry.get("description") or entry.get("definition")
            if isinstance(primary, list):
                primary = ", ".join(str(item) for item in primary)
            if not isinstance(primary, str):
                primary = json.dumps(primary, ensure_ascii=False)

            explanation = entry.get("explanation")
            if isinstance(explanation, list):
                explanation = ", ".join(str(item) for item in explanation)
            if isinstance(explanation, str) and explanation.strip():
                text = f"{primary} — {explanation}"
            else:
                text = primary

            lines.append(f"- {self._truncate_text(text, 140)}")

        return "\n".join(lines) if lines else "No external knowledge provided."

    @staticmethod
    def _truncate_text(text: str, max_length: int) -> str:
        clean_text = " ".join(text.split())
        if len(clean_text) <= max_length:
            return clean_text
        return clean_text[: max_length - 3].rstrip() + "..."

    def _load_schema_text(self, db_id: str) -> str:
        if db_id in self._schema_text_cache:
            return self._schema_text_cache[db_id]

        db_dir = self.describer.dataset_dir / db_id
        schema_path = db_dir / f"{db_id}_schema.txt"
        if not schema_path.exists():
            logger.warning("Schema file missing for %s", db_id)
            schema_text = "Schema not available"
        else:
            try:
                schema_text = schema_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.error("Failed to read schema for %s: %s", db_id, exc)
                schema_text = "Schema not available"

        self._schema_text_cache[db_id] = schema_text
        return schema_text

    def _load_column_meanings(self, db_id: str) -> Dict[str, object]:
        if db_id in self._column_meanings_cache:
            return self._column_meanings_cache[db_id]

        db_dir = self.describer.dataset_dir / db_id
        col_mean_path = db_dir / f"{db_id}_column_meaning_base.json"
        if not col_mean_path.exists():
            logger.warning("Column meaning file missing for %s", db_id)
            meanings: Dict[str, object] = {}
        else:
            try:
                with open(col_mean_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("Failed to read column meanings for %s: %s", db_id, exc)
                raw = {}

            meanings = {key.lower(): value for key, value in raw.items()}

        self._column_meanings_cache[db_id] = meanings
        return meanings

    def _load_external_knowledge(self, db_id: str) -> Sequence[Dict[str, object]]:
        if db_id in self._knowledge_cache:
            return self._knowledge_cache[db_id]

        db_dir = self.describer.dataset_dir / db_id
        kb_path = db_dir / f"{db_id}_kb.jsonl"
        entries: List[Dict[str, object]] = []

        if not kb_path.exists():
            logger.warning("Knowledge base file missing for %s", db_id)
        else:
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            entries.append(json.loads(stripped))
                        except json.JSONDecodeError as exc:
                            logger.debug("Skipping malformed knowledge entry for %s: %s", db_id, exc)
            except OSError as exc:
                logger.error("Failed to read knowledge base for %s: %s", db_id, exc)

        self._knowledge_cache[db_id] = entries
        return entries
