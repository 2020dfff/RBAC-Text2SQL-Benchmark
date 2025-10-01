"""Role processor for LiveSQLBench databases.

This module mirrors the behaviour of :mod:`bird_role_processor` but adapts it
for the structure of the LiveSQLBench SQLite release. It sanitises schema files
(based on ``*_schema.txt``) into temporary ``schema.sql`` files so that the
existing :class:`~src.role_parser.generator.ParallelRoleGenerator` can reuse the
prompting pipeline without modification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.role_parser import ParallelRoleGenerator
from .livesql_describer import LiveSQLDatabaseDescriber, LiveSQLDatabaseInfo

logger = logging.getLogger("livesql_role_assignment")


@dataclass
class LiveSQLDatabaseContext:
    """Container with preprocessed information for a database."""

    name: str
    path: Path
    sqlite_path: Path
    schema_sql: str
    db_info: LiveSQLDatabaseInfo


class LiveSQLRoleProcessor:
    """Process LiveSQLBench databases to generate RBAC role assignments."""

    def __init__(self, generator: ParallelRoleGenerator, describer: LiveSQLDatabaseDescriber):
        self.generator = generator
        self.describer = describer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_databases(
        self,
        db_names: Optional[Iterable[str]] = None,
        batch_size: int = 8,
        run_timestamp: Optional[str] = None,
    ) -> Dict:
        """Process one or more LiveSQL databases and return role assignments."""

        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        database_paths = self._resolve_database_paths(db_names)
        if not database_paths:
            logger.error("No LiveSQL databases found to process")
            return self._empty_result(run_timestamp, 0, batch_size)

        contexts = self._build_database_contexts(database_paths)
        if not contexts:
            logger.error("No valid LiveSQL databases after validation")
            return self._empty_result(run_timestamp, len(database_paths), batch_size)

        logger.info("Preparing to process %d LiveSQL databases", len(contexts))
        for ctx in contexts:
            logger.debug("  - %s", ctx.name)

        total_processed = 0
        total_roles = 0
        assignments: Dict[str, List[Dict]] = {}

        n_batches = (len(contexts) + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch = contexts[batch_start : batch_start + batch_size]
            logger.info("\nProcessing LiveSQL batch %d/%d (%d databases)", batch_idx + 1, n_batches, len(batch))

            db_dirs = [ctx.path for ctx in batch]
            sqlite_paths = {ctx.name: str(ctx.sqlite_path) for ctx in batch}

            temp_files = self._materialise_schema_files(batch)
            try:
                results = self.generator.process_databases_parallel(db_dirs, sqlite_paths=sqlite_paths)
            finally:
                self._cleanup_temp_files(temp_files)

            processed_in_batch = 0
            roles_in_batch = 0
            for result in results:
                if not result or not result.get("roles"):
                    continue
                db_name = result["database"]
                assignments[db_name] = result["roles"]
                processed_in_batch += 1
                roles_in_batch += len(result["roles"])
                logger.info("  ✓ %s: %d roles", db_name, len(result["roles"]))

            total_processed += processed_in_batch
            total_roles += roles_in_batch
            logger.info(
                "Batch %d complete: %d/%d databases produced roles (%d roles)",
                batch_idx + 1,
                processed_in_batch,
                len(batch),
                roles_in_batch,
            )

        return {
            "assignments": assignments,
            "metadata": {
                "timestamp": run_timestamp,
                "total_databases": len(database_paths),
                "processed_databases": total_processed,
                "total_roles_generated": total_roles,
                "batch_size": batch_size,
                "dataset_type": "livesqlbench_base_lite_sqlite",
            },
        }

    def save_role_assignments(self, assignments_data: Dict, output_dir: Path, filename_suffix: Optional[str] = None) -> Path:
        """Persist role assignments to ``output_dir``."""
        if filename_suffix is None:
            filename_suffix = assignments_data.get("metadata", {}).get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"role_assignments_{filename_suffix}_livesql_dev.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(assignments_data, f, ensure_ascii=False, indent=2)
        logger.info("LiveSQL role assignments saved to %s", output_file)
        return output_file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_database_paths(self, db_names: Optional[Iterable[str]]) -> List[Path]:
        if db_names is None:
            return self.describer.list_database_dirs()

        paths: List[Path] = []
        for name in db_names:
            candidate = self.describer.dataset_dir / name
            if candidate.exists() and candidate.is_dir():
                paths.append(candidate)
            else:
                logger.warning("Requested LiveSQL database not found: %s", candidate)
        return sorted(paths)

    def _build_database_contexts(self, db_paths: List[Path]) -> List[LiveSQLDatabaseContext]:
        contexts: List[LiveSQLDatabaseContext] = []
        for db_path in db_paths:
            db_name = db_path.name

            sqlite_path = self._find_sqlite_file(db_path)
            if not sqlite_path:
                logger.warning("Skipping %s: no SQLite template found", db_name)
                continue

            db_info = self.describer.load_database(db_name)
            if not db_info:
                logger.warning("Skipping %s: failed to load schema metadata", db_name)
                continue

            schema_sql = self.describer.build_schema_sql(db_info)
            if not schema_sql.strip():
                logger.warning("Skipping %s: extracted schema SQL is empty", db_name)
                continue

            contexts.append(
                LiveSQLDatabaseContext(
                    name=db_name,
                    path=db_path,
                    sqlite_path=sqlite_path,
                    schema_sql=schema_sql,
                    db_info=db_info,
                )
            )

        return contexts

    @staticmethod
    def _find_sqlite_file(db_path: Path) -> Optional[Path]:
        candidates = sorted(db_path.glob("*.sqlite"))
        return candidates[0] if candidates else None

    @staticmethod
    def _materialise_schema_files(batch: List[LiveSQLDatabaseContext]) -> List[Path]:
        temp_files: List[Path] = []
        for ctx in batch:
            temp_path = ctx.path / "schema.sql"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(ctx.schema_sql)
            temp_files.append(temp_path)
        return temp_files

    @staticmethod
    def _cleanup_temp_files(temp_files: List[Path]) -> None:
        for file_path in temp_files:
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError as exc:
                    logger.warning("Failed to delete temporary schema file %s: %s", file_path, exc)

    @staticmethod
    def _empty_result(timestamp: str, total_databases: int, batch_size: int) -> Dict:
        return {
            "assignments": {},
            "metadata": {
                "timestamp": timestamp,
                "total_databases": total_databases,
                "processed_databases": 0,
                "total_roles_generated": 0,
                "batch_size": batch_size,
                "dataset_type": "livesqlbench_base_lite_sqlite",
            },
        }
