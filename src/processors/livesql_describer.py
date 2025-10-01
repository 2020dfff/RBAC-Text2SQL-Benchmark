"""Utilities for reading and describing LiveSQLBench database schemas.

This module parses the LiveSQLBench SQLite release assets (schema.txt and
column meaning files) and produces structured metadata that can be used to
construct LLM prompts or human-readable documentation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LiveSQLColumnInfo:
    """Metadata for a LiveSQL table column."""

    name: str
    data_type: str
    constraints: Optional[str] = None
    description: Optional[str] = None


@dataclass
class LiveSQLTableInfo:
    """Metadata for a LiveSQL table."""

    name: str
    columns: List[LiveSQLColumnInfo]
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[str] = field(default_factory=list)
    create_statement: str = ""

    @property
    def column_count(self) -> int:
        return len(self.columns)


@dataclass
class LiveSQLDatabaseInfo:
    """Metadata describing an entire LiveSQL database."""

    name: str
    tables: List[LiveSQLTableInfo]

    @property
    def total_tables(self) -> int:
        return len(self.tables)

    @property
    def total_columns(self) -> int:
        return sum(table.column_count for table in self.tables)


class LiveSQLDatabaseDescriber:
    """High level helper for processing LiveSQLBench database assets."""

    COLUMN_KEY_PATTERN = re.compile(r"^(?P<db>[^|]+)\|(?P<table>[^|]+)\|(?P<column>[^|]+)$")

    def __init__(self, project_root: Path, dataset_dir: Optional[Path] = None):
        self.project_root = project_root
        self.dataset_dir = dataset_dir or project_root / "data" / "livesqlbench-base-lite-sqlite"
        self._column_cache: Dict[str, Dict[tuple[str, str], str]] = {}
        self._database_cache: Dict[str, LiveSQLDatabaseInfo] = {}
        self._schema_text_cache: Dict[str, str] = {}
        self._column_prompt_cache: Dict[str, Any] = {}
        self._knowledge_cache: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Public high-level helpers
    # ------------------------------------------------------------------
    def list_database_dirs(self) -> List[Path]:
        """Return all LiveSQL database directories within the dataset folder."""
        if not self.dataset_dir.exists():
            logger.warning("LiveSQL dataset directory not found: %s", self.dataset_dir)
            return []
        return sorted([p for p in self.dataset_dir.iterdir() if p.is_dir()])

    def load_database(self, db_name: str) -> Optional[LiveSQLDatabaseInfo]:
        """Load metadata for a database, reading from disk if necessary."""
        db_name = db_name.strip()
        if db_name in self._database_cache:
            return self._database_cache[db_name]

        db_path = self.dataset_dir / db_name
        if not db_path.exists() or not db_path.is_dir():
            logger.error("LiveSQL database directory missing: %s", db_path)
            return None

        schema_path = db_path / f"{db_name}_schema.txt"
        if not schema_path.exists():
            logger.error("Schema file missing for %s: %s", db_name, schema_path)
            return None

        column_meanings = self._load_column_meanings(db_name)
        tables = self._parse_schema_file(schema_path, column_meanings)

        if not tables:
            logger.error("Failed to parse any tables for database %s", db_name)
            return None

        db_info = LiveSQLDatabaseInfo(name=db_name, tables=tables)
        self._database_cache[db_name] = db_info
        return db_info

    def get_prompt_assets(
        self, db_name: str
    ) -> tuple[str, Any, List[Dict[str, Any]]]:
        """Return raw schema text, column meanings JSON, and knowledge entries for prompts."""

        schema_text = self._load_schema_text(db_name)
        column_meanings = self._load_column_meanings_json(db_name)
        knowledge_entries = self._load_external_knowledge(db_name)
        return schema_text, column_meanings, knowledge_entries

    def generate_schema_description(
        self,
        db_info: LiveSQLDatabaseInfo,
        include_constraints: bool = True,
        max_description_length: int = 400,
    ) -> str:
        """Produce a detailed schema description suitable for LLM prompts."""
        if not db_info or not db_info.tables:
            return f"Database {db_info.name if db_info else 'unknown'} has no schema information available."

        lines: List[str] = [
            f"Database: {db_info.name}",
            f"Total Tables: {db_info.total_tables}",
            f"Total Columns: {db_info.total_columns}",
            "",
            "Table Schemas:",
        ]

        for table in db_info.tables:
            header = f"\nTable: {table.name} ({table.column_count} columns)"
            separator = "-" * len(header.lstrip())
            lines.append(header)
            lines.append(separator)

            for column in table.columns:
                description = column.description or ""
                if description:
                    description = self._truncate_description(description, max_description_length)

                bullet = f"• {column.name} ({column.data_type})"
                if description:
                    bullet += f": {description}"
                elif include_constraints and column.constraints:
                    bullet += f" — {column.constraints}"
                lines.append(bullet)

                if include_constraints and column.constraints and description:
                    lines.append(f"  Constraints: {column.constraints}")

            if include_constraints and table.primary_keys:
                pk_text = ", ".join(table.primary_keys)
                lines.append(f"Primary Key: {pk_text}")

            if include_constraints and table.foreign_keys:
                lines.append("Foreign Keys:")
                for fk in table.foreign_keys:
                    lines.append(f"  - {fk}")

        return "\n".join(lines)

    def build_schema_sql(self, db_info: LiveSQLDatabaseInfo) -> str:
        """Construct sanitized CREATE TABLE statements for a database."""
        statements = []
        for table in db_info.tables:
            stmt = table.create_statement.strip()
            if not stmt.endswith(";"):
                stmt = f"{stmt};"
            statements.append(stmt)
        return "\n\n".join(statements)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_column_meanings(self, db_name: str) -> Dict[tuple[str, str], str]:
        if db_name in self._column_cache:
            return self._column_cache[db_name]

        column_file = self.dataset_dir / db_name / f"{db_name}_column_meaning_base.json"
        column_map: Dict[tuple[str, str], str] = {}

        if not column_file.exists():
            logger.warning("Column meaning file not found for %s", db_name)
            self._column_cache[db_name] = column_map
            return column_map

        try:
            with open(column_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode column meaning file %s: %s", column_file, exc)
            self._column_cache[db_name] = column_map
            return column_map

        for key, description in data.items():
            match = self.COLUMN_KEY_PATTERN.match(key)
            if not match:
                continue
            table_key = match.group("table").lower()
            column_key = match.group("column").lower()
            column_map[(table_key, column_key)] = self._normalise_column_description(description)

        self._column_cache[db_name] = column_map
        return column_map

    def _load_column_meanings_json(self, db_name: str) -> Any:
        if db_name in self._column_prompt_cache:
            return self._column_prompt_cache[db_name]

        column_file = self.dataset_dir / db_name / f"{db_name}_column_meaning_base.json"
        if not column_file.exists():
            logger.warning("Column meaning file not found for prompt assets: %s", column_file)
            self._column_prompt_cache[db_name] = {}
            return {}

        try:
            with open(column_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode column meaning JSON for prompt: %s", exc)
            payload = {}

        self._column_prompt_cache[db_name] = payload
        return payload

    def _normalise_column_description(self, description: Any) -> str:
        if isinstance(description, str):
            return description.strip()

        if isinstance(description, dict):
            parts: List[str] = []
            column_meaning = description.get("column_meaning")
            if isinstance(column_meaning, str):
                parts.append(column_meaning.strip())

            fields_meaning = description.get("fields_meaning")
            if isinstance(fields_meaning, dict):
                field_chunks = []
                for field, meaning in fields_meaning.items():
                    if isinstance(meaning, str):
                        field_chunks.append(f"{field}: {meaning.strip()}")
                if field_chunks:
                    parts.append("Fields - " + "; ".join(field_chunks))

            if parts:
                return " ".join(parts)

            return json.dumps(description, ensure_ascii=False)

        return str(description)

    def _parse_schema_file(
        self,
        schema_path: Path,
        column_meanings: Dict[tuple[str, str], str],
    ) -> List[LiveSQLTableInfo]:
        tables: List[LiveSQLTableInfo] = []
        current_lines: List[str] = []
        current_name: Optional[str] = None

        with open(schema_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            raw_line = lines[i]
            line = raw_line.strip()

            if not current_lines and line.upper().startswith("CREATE TABLE"):
                current_name = self._extract_table_name(line)
                current_lines.append(raw_line.rstrip("\n"))
                i += 1
                # Capture until we hit the closing ");"
                while i < len(lines):
                    current_lines.append(lines[i].rstrip("\n"))
                    if lines[i].strip().startswith(");"):
                        break
                    i += 1

                if current_name:
                    table = self._build_table_info(current_name, current_lines, column_meanings)
                    tables.append(table)
                else:
                    logger.warning(
                        "Unable to parse table name from CREATE TABLE statement starting with: %s",
                        line,
                    )

                # Reset for next table
                current_lines = []
                current_name = None
            i += 1

        return tables

    def _build_table_info(
        self,
        table_name: str,
        create_lines: List[str],
        column_meanings: Dict[tuple[str, str], str],
    ) -> LiveSQLTableInfo:
        column_infos: List[LiveSQLColumnInfo] = []
        primary_keys: List[str] = []
        foreign_keys: List[str] = []

        # Exclude the first "CREATE TABLE" line and the final closing line when parsing columns
        body_lines = create_lines[1:-1]
        for raw_line in body_lines:
            stripped = raw_line.strip().rstrip(",")
            if not stripped:
                continue

            upper = stripped.upper()
            if upper.startswith("PRIMARY KEY"):
                primary_keys.extend(self._extract_identifiers(stripped))
                continue
            if upper.startswith("FOREIGN KEY"):
                fk_desc = self._format_foreign_key(stripped)
                if fk_desc:
                    foreign_keys.append(fk_desc)
                continue

            column_info = self._parse_column_definition(stripped, table_name, column_meanings)
            if column_info:
                column_infos.append(column_info)

        create_statement = "\n".join(create_lines).strip()
        return LiveSQLTableInfo(
            name=table_name,
            columns=column_infos,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            create_statement=create_statement,
        )

    def _parse_column_definition(
        self,
        line: str,
        table_name: str,
        column_meanings: Dict[tuple[str, str], str],
    ) -> Optional[LiveSQLColumnInfo]:
        match = re.match(r'"?(?P<name>[A-Za-z0-9_]+)"?\s+(?P<rest>.+)', line)
        if not match:
            logger.debug("Skipping unparsable column definition: %s", line)
            return None

        column_name = match.group("name").strip()
        rest = match.group("rest").strip()

        # Extract datatype (token + optional parenthesised segment)
        type_match = re.match(r"(?P<dtype>[A-Za-z0-9_]+(?:\([^)]*\))?)\s*(?P<constraints>.*)", rest)
        if not type_match:
            data_type = rest
            constraints = ""
        else:
            data_type = type_match.group("dtype").strip()
            constraints = type_match.group("constraints").strip()

        desc_key = (table_name.lower(), column_name.lower())
        description = column_meanings.get(desc_key)

        if description:
            description = self._clean_description(description)
        if constraints:
            constraints = constraints.rstrip(",")

        return LiveSQLColumnInfo(
            name=column_name,
            data_type=data_type,
            constraints=constraints or None,
            description=description,
        )

    def _load_schema_text(self, db_name: str) -> str:
        if db_name in self._schema_text_cache:
            return self._schema_text_cache[db_name]

        schema_path = self.dataset_dir / db_name / f"{db_name}_schema.txt"
        if not schema_path.exists():
            logger.warning("Schema file missing for prompt assets: %s", schema_path)
            schema_text = ""
        else:
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema_text = f.read()
            except OSError as exc:
                logger.error("Failed to read schema text for %s: %s", db_name, exc)
                schema_text = ""

        self._schema_text_cache[db_name] = schema_text
        return schema_text

    def _load_external_knowledge(self, db_name: str) -> List[Dict[str, Any]]:
        if db_name in self._knowledge_cache:
            return self._knowledge_cache[db_name]

        kb_path = self.dataset_dir / db_name / f"{db_name}_kb.jsonl"
        entries: List[Dict[str, Any]] = []

        if not kb_path.exists():
            logger.debug("Knowledge file not found for %s: %s", db_name, kb_path)
            self._knowledge_cache[db_name] = entries
            return entries

        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        visible_fields = {key: payload.get(key) for key in ("id", "knowledge", "description", "definition") if key in payload}
                        if visible_fields:
                            entries.append(visible_fields)
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed knowledge line for %s", db_name)
        except OSError as exc:
            logger.error("Failed to read knowledge file for %s: %s", db_name, exc)

        self._knowledge_cache[db_name] = entries
        return entries

    @staticmethod
    def _clean_description(description: str) -> str:
        """Normalize spacing of long description strings."""
        text = " ".join(description.split())
        # Remove redundant prefixes that sometimes appear in the JSON
        replacements = {
            "Full name:": "",
            "Explanation:": "",
            "Data type:": "",
        }
        for prefix, replacement in replacements.items():
            text = text.replace(prefix, replacement)
        return text.strip()

    @staticmethod
    def _truncate_description(description: str, max_length: int = 400) -> str:
        if len(description) <= max_length:
            return description
        return description[: max_length - 3].rstrip() + "..."

    @staticmethod
    def _extract_table_name(create_line: str) -> Optional[str]:
        match = re.match(r"CREATE TABLE\s+\"?(?P<name>[A-Za-z0-9_]+)\"?", create_line, re.IGNORECASE)
        if match:
            return match.group("name")
        return None

    @staticmethod
    def _extract_identifiers(clause: str) -> List[str]:
        identifiers = re.findall(r"\w+", clause)
        # Remove SQL keywords
        keywords = {"PRIMARY", "KEY", "FOREIGN", "REFERENCES"}
        return [ident for ident in identifiers if ident.upper() not in keywords]

    @staticmethod
    def _format_foreign_key(clause: str) -> Optional[str]:
        match = re.match(
            r"FOREIGN KEY\s*\((?P<column>[^)]+)\)\s*REFERENCES\s+\"?(?P<table>[A-Za-z0-9_]+)\"?\s*\((?P<target>[^)]+)\)",
            clause,
            re.IGNORECASE,
        )
        if not match:
            return None
        column = match.group("column").strip()
        table = match.group("table").strip()
        target = match.group("target").strip()
        return f"{column} -> {table}({target})"

