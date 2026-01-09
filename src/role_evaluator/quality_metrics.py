"""
Role Generation Quality Metrics

This module provides reusable quality assessment utilities for evaluating
role generation quality in RBAC datasets. It includes:

1. Deny Rate Analysis - Evaluate permission distribution balance
2. Semantic Similarity Analysis - Embedding-based role-schema alignment
3. Policy Overlap Analysis - Jaccard similarity between role policies

Supports TWO permission granularity levels:
- **Column-Level RBAC**: SELECT permission on specific columns (Bird, Spider)
- **CRUD-Level RBAC**: DDL, INSERT, DELETE, UPDATE permissions (LiveSQLBench-Full)

Usage (Column-Level):
    from src.role_evaluator.quality_metrics import RoleQualityEvaluator
    
    evaluator = RoleQualityEvaluator(
        role_assignments_path="outputs/column_level_role_assignments_bird.json",
        dataset_path="outputs/column_level_rbac_dataset_bird.json",
        schema_dir="data/Bird/dev_20251106/dev_databases"
    )
    report = evaluator.evaluate_all()

Usage (CRUD-Level):
    from src.role_evaluator.quality_metrics import CrudRoleQualityEvaluator
    
    evaluator = CrudRoleQualityEvaluator(
        role_assignments_path="outputs/crud_role_assignments_livesqlbench.json",
        dataset_path="outputs/crud_rbac_dataset_livesqlbench.json"
    )
    report = evaluator.evaluate_all()
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import re
import numpy as np


@dataclass
class DenyRateMetrics:
    """Metrics for deny rate analysis per database."""
    db_id: str
    total_entries: int
    allowed_count: int
    denied_count: int
    deny_rate: float
    all_allow_queries: int
    all_deny_queries: int  # queries where ALL roles deny (should be 0)
    total_queries: int
    all_allow_pct: float
    all_deny_pct: float  # should be 0% if SystemManager is working
    roles: List[str]
    issues: List[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return len(self.issues) == 0


@dataclass
class SemanticMetrics:
    """Metrics for semantic quality per database.
    
    Measures how well role names/descriptions align with schema content.
    This is purely about semantic alignment, NOT about permission coverage.
    """
    db_id: str
    avg_similarity: float  # Average cosine similarity between role text and schema
    num_roles: int
    min_similarity: float = 0.0  # Minimum similarity (worst aligned role)
    max_similarity: float = 1.0  # Maximum similarity (best aligned role)
    avg_coverage: float = 0.0  # Average column coverage (excluding SystemManager)
    coverage_std: float = 0.0  # Standard deviation of coverage
    issues: List[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return len(self.issues) == 0


@dataclass
class OverlapMetrics:
    """Metrics for policy overlap per database."""
    db_id: str
    avg_overlap: float
    max_overlap: float
    max_pair: Tuple[str, str]
    num_pairs: int
    issues: List[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return len(self.issues) == 0


# ============================================================================
# CRUD-Level Specific Dataclasses
# ============================================================================

@dataclass
class CrudDenyRateMetrics:
    """Deny rate metrics for CRUD-level RBAC per database.
    
    CRUD permissions: DDL (CREATE/DROP/ALTER), INSERT, DELETE, UPDATE
    """
    db_id: str
    total_entries: int
    allowed_count: int
    denied_count: int
    deny_rate: float
    
    # Per-operation breakdown
    ddl_allowed: int = 0
    ddl_denied: int = 0
    insert_allowed: int = 0
    insert_denied: int = 0
    delete_allowed: int = 0
    delete_denied: int = 0
    update_allowed: int = 0
    update_denied: int = 0
    select_allowed: int = 0  # For SELECT queries if any
    select_denied: int = 0
    
    # Query-level metrics
    all_allow_queries: int = 0
    all_deny_queries: int = 0
    total_queries: int = 0
    all_allow_pct: float = 0.0
    all_deny_pct: float = 0.0
    
    roles: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return len(self.issues) == 0
    
    @property
    def ddl_deny_rate(self) -> float:
        total = self.ddl_allowed + self.ddl_denied
        return (self.ddl_denied / total * 100) if total > 0 else 0.0
    
    @property
    def insert_deny_rate(self) -> float:
        total = self.insert_allowed + self.insert_denied
        return (self.insert_denied / total * 100) if total > 0 else 0.0
    
    @property
    def delete_deny_rate(self) -> float:
        total = self.delete_allowed + self.delete_denied
        return (self.delete_denied / total * 100) if total > 0 else 0.0
    
    @property
    def update_deny_rate(self) -> float:
        total = self.update_allowed + self.update_denied
        return (self.update_denied / total * 100) if total > 0 else 0.0
    
    @property
    def select_deny_rate(self) -> float:
        total = self.select_allowed + self.select_denied
        return (self.select_denied / total * 100) if total > 0 else 0.0
    
    # Total counts for each operation
    @property
    def ddl_total(self) -> int:
        return self.ddl_allowed + self.ddl_denied
    
    @property
    def insert_total(self) -> int:
        return self.insert_allowed + self.insert_denied
    
    @property
    def delete_total(self) -> int:
        return self.delete_allowed + self.delete_denied
    
    @property
    def update_total(self) -> int:
        return self.update_allowed + self.update_denied
    
    @property
    def select_total(self) -> int:
        return self.select_allowed + self.select_denied


@dataclass
class CrudCoverageMetrics:
    """Coverage metrics for CRUD-level RBAC per database.
    
    Coverage is the AVERAGE per-role coverage (not union of all roles).
    - INSERT/DELETE: table-level (% of schema tables each role can access)
    - SELECT/UPDATE: column-level (% of schema columns each role can access)
    
    Diversity (coverage_std) measures how different roles' permissions are.
    Per-operation std allows fine-grained analysis of role differentiation.
    """
    db_id: str
    
    # Schema totals (ground truth)
    total_tables: int = 0
    total_columns: int = 0
    
    # Average per-role coverage (excludes SystemManager)
    avg_insert_coverage: float = 0.0  # Avg % of tables each role can INSERT
    avg_delete_coverage: float = 0.0  # Avg % of tables each role can DELETE
    avg_select_coverage: float = 0.0  # Avg % of columns each role can SELECT
    avg_update_coverage: float = 0.0  # Avg % of columns each role can UPDATE
    
    # DDL is database-level
    has_ddl_role: bool = False  # Any role has DDL=True
    
    # Role diversity - per-operation std (new)
    num_roles: int = 0
    avg_permissions_per_role: float = 0.0  # Average column-level permissions per role
    
    # Per-operation coverage std (measures diversity within each operation type)
    select_coverage_std: float = 0.0  # Std of SELECT coverage ratios across roles
    update_coverage_std: float = 0.0  # Std of UPDATE coverage ratios across roles
    insert_coverage_std: float = 0.0  # Std of INSERT coverage ratios across roles (table-level)
    delete_coverage_std: float = 0.0  # Std of DELETE coverage ratios across roles (table-level)
    
    # Weighted overall std (SELECT-weighted since it's the primary differentiator)
    coverage_std: float = 0.0  # Backward compatible: weighted combination
    
    issues: List[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return len(self.issues) == 0
    
    @property
    def insert_coverage(self) -> float:
        """Average INSERT coverage per role (table level)."""
        return self.avg_insert_coverage
    
    @property
    def delete_coverage(self) -> float:
        """Average DELETE coverage per role (table level)."""
        return self.avg_delete_coverage
    
    @property
    def select_coverage(self) -> float:
        """Average SELECT coverage per role (column level)."""
        return self.avg_select_coverage
    
    @property
    def update_coverage(self) -> float:
        """Average UPDATE coverage per role (column level)."""
        return self.avg_update_coverage


@dataclass
class CrudQualityReport:
    """Comprehensive quality report for CRUD-level role generation."""
    deny_rate_metrics: Dict[str, CrudDenyRateMetrics] = field(default_factory=dict)
    coverage_metrics: Dict[str, CrudCoverageMetrics] = field(default_factory=dict)
    overlap_metrics: Dict[str, OverlapMetrics] = field(default_factory=dict)
    semantic_metrics: Dict[str, SemanticMetrics] = field(default_factory=dict)  # Role-schema alignment
    semantic_role_details: List[Dict] = field(default_factory=list)  # Per-role semantic data for visualization
    
    # Summary statistics
    total_databases: int = 0
    total_roles: int = 0
    total_entries: int = 0
    
    # Issues summary
    databases_with_issues: List[str] = field(default_factory=list)
    all_issues: Dict[str, List[str]] = field(default_factory=dict)
    
    # Operation distribution summary
    operation_distribution: Dict[str, int] = field(default_factory=dict)
    
    @property
    def pass_rate(self) -> float:
        if self.total_databases == 0:
            return 0.0
        return (self.total_databases - len(self.databases_with_issues)) / self.total_databases


@dataclass
class QualityReport:
    """Comprehensive quality report for role generation."""
    deny_rate_metrics: Dict[str, DenyRateMetrics] = field(default_factory=dict)
    semantic_metrics: Dict[str, SemanticMetrics] = field(default_factory=dict)
    overlap_metrics: Dict[str, OverlapMetrics] = field(default_factory=dict)
    
    # Summary statistics
    total_databases: int = 0
    total_roles: int = 0
    total_entries: int = 0
    
    # Issues summary
    databases_with_issues: List[str] = field(default_factory=list)
    all_issues: Dict[str, List[str]] = field(default_factory=dict)
    
    @property
    def pass_rate(self) -> float:
        if self.total_databases == 0:
            return 0.0
        return (self.total_databases - len(self.databases_with_issues)) / self.total_databases


class RoleQualityEvaluator:
    """
    Comprehensive evaluator for role generation quality.
    
    Evaluates:
    - Deny rate distribution (should be 5%-90%)
    - Semantic alignment between roles and schemas
    - Policy overlap between roles (should be <70%)
    
    Note on All-Allow Rate:
        A low all-allow rate is NOT necessarily bad. For complex databases with
        fine-grained column-level permissions, it's expected that many queries
        will be denied because they require columns outside the role's access.
        This is the intended behavior of column-level RBAC.
    
    Note on Policy Overlap (Max Overlap):
        Max Overlap = highest Jaccard similarity between any two role policies.
        Jaccard(A, B) = |A ∩ B| / |A ∪ B|, where A and B are sets of table.column.
        
        High overlap (>70%) indicates:
        - Redundant roles: Two roles grant nearly identical column access
        - Low role diversity: Roles don't differentiate access patterns well
        - Training inefficiency: Similar roles don't add discriminative signal
        
        Example: If RoleA accesses {t1.c1, t1.c2, t2.c1} and RoleB accesses
        {t1.c1, t1.c2, t2.c2}, overlap = 2/4 = 50%. If >70%, they're too similar.
    """
    
    # Quality thresholds (class defaults, can be overridden in __init__)
    DENY_RATE_MIN = 5.0    # Minimum acceptable deny rate (%)
    DENY_RATE_MAX = 90.0   # Maximum acceptable deny rate (%)
    # Note: We don't flag high all-allow as an issue - complex DBs naturally have lower all-allow
    SIMILARITY_MIN = 0.6   # Minimum semantic similarity threshold
    OVERLAP_MAX = 0.7      # Maximum acceptable policy overlap (Jaccard similarity)
    COVERAGE_STD_MIN = 0.1 # Minimum coverage std dev for diversity
    COVERAGE_MAX = 0.9     # Maximum acceptable average coverage
    
    def __init__(
        self,
        role_assignments_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        schema_dir: Optional[str] = None,
        role_assignments: Optional[Dict] = None,
        dataset: Optional[List[Dict]] = None,
        schema_summaries: Optional[Dict] = None,
        embed_model: str = "text-embedding-3-small",
        embed_backend: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize evaluator with file paths or pre-loaded data.
        
        Args:
            role_assignments_path: Path to role assignments JSON file
            dataset_path: Path to RBAC dataset JSON file
            schema_dir: Path to database schema directory (with *_schema_detailed.txt files)
            role_assignments: Pre-loaded role assignments dict
            dataset: Pre-loaded dataset list
            schema_summaries: Pre-loaded schema summaries dict
            embed_model: Embedding model name (default: text-embedding-3-small)
                - OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
                - BAAI/fastembed: "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"
            embed_backend: 'openai' or 'fastembed' (auto-detected if None)
            thresholds: Dict of quality thresholds to override defaults:
                - deny_rate_min: Minimum deny rate (default: 5.0)
                - deny_rate_max: Maximum deny rate (default: 90.0)
                - coverage_max: Maximum average coverage (default: 0.9)
                - coverage_std_min: Minimum coverage std (default: 0.1)
                - max_overlap: Maximum policy overlap (default: 0.7)
        """
        self.embed_model = embed_model
        self.embed_backend = embed_backend
        self.role_assignments = role_assignments
        self.dataset = dataset
        self.schema_summaries = schema_summaries
        
        # Override thresholds if provided
        if thresholds:
            if "deny_rate_min" in thresholds:
                self.DENY_RATE_MIN = thresholds["deny_rate_min"]
            if "deny_rate_max" in thresholds:
                self.DENY_RATE_MAX = thresholds["deny_rate_max"]
            if "coverage_max" in thresholds:
                self.COVERAGE_MAX = thresholds["coverage_max"]
            if "coverage_std_min" in thresholds:
                self.COVERAGE_STD_MIN = thresholds["coverage_std_min"]
            if "max_overlap" in thresholds:
                self.OVERLAP_MAX = thresholds["max_overlap"]
        
        # Load from files if provided
        if role_assignments_path and not self.role_assignments:
            self._load_role_assignments(role_assignments_path)
        
        if dataset_path and not self.dataset:
            self._load_dataset(dataset_path)
        
        if schema_dir and not self.schema_summaries:
            self._load_schema_summaries(schema_dir)
        
        # Embedding cache
        self._embeddings_computed = False
        self._role_emb_dict = {}
        self._schema_emb_dict = {}
    
    def _load_role_assignments(self, path: str):
        """Load role assignments from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        # Handle both formats: {"assignments": {...}} or direct dict
        self.role_assignments = data.get("assignments", data)
    
    def _load_dataset(self, path: str):
        """Load RBAC dataset from JSON file."""
        with open(path, 'r') as f:
            self.dataset = json.load(f)
    
    def _load_schema_summaries(self, schema_dir: str):
        """Load schema summaries from various sources.
        
        Supports:
        - Bird: *_schema_detailed.txt files
        - Spider: schema.sql files or SQLite databases
        """
        schema_path = Path(schema_dir)
        self.schema_summaries = {}
        
        for db_dir in sorted(schema_path.iterdir()):
            if not db_dir.is_dir() or db_dir.name.startswith('.'):
                continue
            
            db_id = db_dir.name
            table_columns = {}  # {table_name: column_count}
            table_count = 0
            column_count = 0
            schema_text = ""
            
            # Try Bird format first: *_schema_detailed.txt
            schema_files = list(db_dir.glob('*_schema_detailed.txt'))
            
            if schema_files:
                schema_text = schema_files[0].read_text(encoding='utf-8')
                
                # Parse total table and column counts
                match = re.search(r'Tables:\s*(\d+),\s*Columns:\s*(\d+)', schema_text)
                table_count = int(match.group(1)) if match else 0
                column_count = int(match.group(2)) if match else 0
                
                # Parse per-table column counts: "Table: xxx (N columns)"
                for table_match in re.finditer(r'Table:\s+(\w+)\s+\((\d+)\s+columns?\)', schema_text):
                    table_name = table_match.group(1)
                    cols = int(table_match.group(2))
                    table_columns[table_name] = cols
            
            # Try Spider format: schema.sql (parse CREATE TABLE statements)
            if not table_columns:
                schema_sql_file = db_dir / "schema.sql"
                if schema_sql_file.exists():
                    try:
                        schema_text = schema_sql_file.read_text(encoding='utf-8')
                        # Parse CREATE TABLE statements
                        table_columns = self._parse_schema_sql(schema_text)
                        table_count = len(table_columns)
                        column_count = sum(table_columns.values())
                    except Exception:
                        pass
            
            # Final fallback: query SQLite database directly
            if not table_columns:
                db_file = db_dir / f"{db_id}.sqlite"
                if db_file.exists():
                    try:
                        import sqlite3
                        conn = sqlite3.connect(str(db_file))
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                        tables = [r[0] for r in cursor.fetchall()]
                        
                        # Generate synthetic schema text for semantic embedding
                        schema_lines = [f"Database: {db_id}", ""]
                        
                        for table in tables:
                            cursor.execute(f"PRAGMA table_info(`{table}`)")
                            cols = cursor.fetchall()
                            table_columns[table] = len(cols)
                            
                            # Build semantic-rich schema text
                            col_details = []
                            for col in cols:
                                # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
                                col_name = col[1]
                                col_type = col[2] if col[2] else "TEXT"
                                is_pk = " PRIMARY KEY" if col[5] else ""
                                col_details.append(f"  {col_name} {col_type}{is_pk}")
                            
                            schema_lines.append(f"CREATE TABLE {table} (")
                            schema_lines.append(",\n".join(col_details))
                            schema_lines.append(");")
                            schema_lines.append("")
                        
                        conn.close()
                        table_count = len(tables)
                        column_count = sum(table_columns.values())
                        
                        # Use generated schema text for embedding
                        schema_text = "\n".join(schema_lines)
                    except Exception:
                        pass
            
            self.schema_summaries[db_id] = {
                "db_id": db_id,
                "table_count": table_count,
                "column_count": column_count,
                "table_columns": table_columns,  # Per-table column counts
                "summary_text": schema_text,
            }
    
    def _parse_schema_sql(self, schema_text: str) -> Dict[str, int]:
        """Parse CREATE TABLE statements from schema.sql to get table column counts."""
        table_columns = {}
        
        # Pattern to match CREATE TABLE statements
        table_pattern = re.compile(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?([A-Za-z0-9_]+)[`\"\]]?",
            re.IGNORECASE
        )
        
        table_matches = list(table_pattern.finditer(schema_text))
        
        for i, match in enumerate(table_matches):
            table_name = match.group(1).strip('`"[]').lower()
            
            # Find the block between this CREATE TABLE and the next (or end)
            start_pos = match.end()
            if i + 1 < len(table_matches):
                end_pos = table_matches[i + 1].start()
            else:
                end_pos = len(schema_text)
            
            block = schema_text[start_pos:end_pos]
            
            # Find opening parenthesis
            paren_start = block.find('(')
            if paren_start == -1:
                continue
            
            # Find matching closing paren
            depth = 0
            paren_end = -1
            for j, char in enumerate(block[paren_start:], paren_start):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        paren_end = j
                        break
            
            if paren_end == -1:
                continue
            
            columns_block = block[paren_start+1:paren_end]
            
            # Count column definitions (excluding constraints)
            column_count = 0
            for line in columns_block.split(','):
                line = line.strip()
                if not line:
                    continue
                
                # Skip constraints
                upper_line = line.upper()
                if any(kw in upper_line for kw in ['PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'CONSTRAINT']):
                    continue
                
                # Check if it looks like a column definition
                col_match = re.match(r'^[`\"\[]?([A-Za-z0-9_]+)[`\"\]]?\s+', line)
                if col_match:
                    col_name = col_match.group(1).upper()
                    # Validate it's not a keyword
                    if col_name not in ['PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'CONSTRAINT', 'INDEX', 'KEY']:
                        column_count += 1
            
            if column_count > 0:
                table_columns[table_name] = column_count
        
        return table_columns
    
    def evaluate_deny_rates(self) -> Dict[str, DenyRateMetrics]:
        """
        Evaluate deny rate distribution across databases.
        
        Checks:
        - Overall deny rate (should be 5%-90%)
        - All-allow query percentage (should be <50%)
        - All-deny queries (should be 0 - SystemManager coverage)
        - Zero deny databases (problematic for training)
        
        Returns:
            Dict mapping db_id to DenyRateMetrics
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Provide dataset_path or dataset.")
        
        metrics = {}
        
        # Group entries by database
        db_stats = defaultdict(lambda: {'allowed': 0, 'denied': 0, 'roles': set()})
        for entry in self.dataset:
            db_id = entry['db_id']
            db_stats[db_id][entry['metadata']['permission']] += 1
            db_stats[db_id]['roles'].add(entry['role'])
        
        # Analyze query-level patterns
        query_groups = defaultdict(list)
        for entry in self.dataset:
            question_part = entry['input'].split('\n\nSchema:')[0][:200] if '\n\nSchema:' in entry['input'] else entry['input'][:200]
            key = (entry['db_id'], question_part, entry['metadata']['gold_sql'])
            query_groups[key].append(entry)
        
        # Count all-allow and all-deny queries per database
        all_allow_by_db = defaultdict(int)
        all_deny_by_db = defaultdict(int)
        query_count_by_db = defaultdict(int)
        
        for key, entries in query_groups.items():
            db_id = key[0]
            query_count_by_db[db_id] += 1
            permissions = [e['metadata']['permission'] for e in entries]
            if all(p == 'allowed' for p in permissions):
                all_allow_by_db[db_id] += 1
            if all(p == 'denied' for p in permissions):
                all_deny_by_db[db_id] += 1
        
        # Build metrics for each database
        for db_id in sorted(db_stats.keys()):
            stats = db_stats[db_id]
            total = stats['allowed'] + stats['denied']
            deny_rate = stats['denied'] / total * 100 if total > 0 else 0
            
            total_queries = query_count_by_db[db_id]
            all_allow_queries = all_allow_by_db[db_id]
            all_deny_queries = all_deny_by_db[db_id]
            all_allow_pct = all_allow_queries / total_queries * 100 if total_queries > 0 else 0
            all_deny_pct = all_deny_queries / total_queries * 100 if total_queries > 0 else 0
            
            issues = []
            if deny_rate < self.DENY_RATE_MIN:
                issues.append(f"deny rate too low ({deny_rate:.1f}% < {self.DENY_RATE_MIN}%)")
            if deny_rate > self.DENY_RATE_MAX:
                issues.append(f"deny rate too high ({deny_rate:.1f}% > {self.DENY_RATE_MAX}%)")
            # Note: We don't flag low all-allow rate as an issue.
            # Complex databases with fine-grained column permissions naturally have 
            # lower all-allow rates - queries often need columns outside role access.
            if all_deny_queries > 0:
                issues.append(f"all-deny queries found ({all_deny_queries}) - SystemManager coverage broken")
            
            metrics[db_id] = DenyRateMetrics(
                db_id=db_id,
                total_entries=total,
                allowed_count=stats['allowed'],
                denied_count=stats['denied'],
                deny_rate=deny_rate,
                all_allow_queries=all_allow_queries,
                all_deny_queries=all_deny_queries,
                total_queries=total_queries,
                all_allow_pct=all_allow_pct,
                all_deny_pct=all_deny_pct,
                roles=list(stats['roles']),
                issues=issues
            )
        
        return metrics
    
    def _compute_embeddings(self):
        """Compute embeddings for role texts and schema texts."""
        if self._embeddings_computed:
            return
        
        # Import embedding utilities
        try:
            from plot.embedding_utils import embed_texts
        except ImportError:
            # Fallback: try to import from project root
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root / "plot"))
            from embedding_utils import embed_texts
        
        # Use configured embedding model and backend
        EMBED_MODEL = self.embed_model
        EMBED_BACKEND = self.embed_backend  # None = auto-detect
        
        # Get API key from environment (only needed for OpenAI backend)
        import os
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        # Prepare role records
        role_records = self._prepare_role_records()
        
        # Get unique texts
        unique_schema_texts = list(set(
            spec["schema_text"] for spec in role_records if spec["schema_text"]
        ))
        unique_role_texts = list(set(spec["role_text"] for spec in role_records))
        
        # Compute embeddings (backend auto-detected based on model name if not specified)
        if unique_schema_texts:
            schema_embeddings = embed_texts(
                unique_schema_texts, 
                model_name=EMBED_MODEL, 
                backend=EMBED_BACKEND,
                openai_api_key=openai_api_key
            )
            self._schema_emb_dict = {
                text: emb for text, emb in zip(unique_schema_texts, schema_embeddings)
            }
        
        if unique_role_texts:
            role_embeddings = embed_texts(
                unique_role_texts, 
                model_name=EMBED_MODEL, 
                backend=EMBED_BACKEND,
                openai_api_key=openai_api_key
            )
            self._role_emb_dict = {
                text: emb for text, emb in zip(unique_role_texts, role_embeddings)
            }
        
        self._embeddings_computed = True
    
    def _prepare_role_records(self) -> List[Dict]:
        """Prepare role records for semantic analysis."""
        if not self.role_assignments:
            raise ValueError("Role assignments not loaded.")
        
        role_specs = []
        
        for db_id, roles_list in self.role_assignments.items():
            schema_summary = self.schema_summaries.get(db_id, {}) if self.schema_summaries else {}
            schema_text = schema_summary.get("summary_text", "")
            total_tables = schema_summary.get("table_count", 0)
            total_columns = schema_summary.get("column_count", 0)
            table_columns = schema_summary.get("table_columns", {})  # {table_name: col_count}
            
            for role in roles_list:
                role_name = role.get("role", "")
                description = role.get("description", "")
                policy = role.get("policy", {})
                
                # Count accessible tables and columns
                accessible_tables = len(policy)
                accessible_columns = 0
                
                for table, cols in policy.items():
                    if cols == ["*"] or cols == "*":
                        # "*" means all columns in THIS table
                        # Look up actual column count for this table
                        # Case-insensitive lookup
                        table_col_count = table_columns.get(table)
                        if table_col_count is None:
                            # Try case-insensitive match
                            for t_name, t_cols in table_columns.items():
                                if t_name.lower() == table.lower():
                                    table_col_count = t_cols
                                    break
                        if table_col_count:
                            accessible_columns += table_col_count
                        # else: unknown table, skip (will be 0)
                    elif isinstance(cols, list):
                        accessible_columns += len(cols)
                    else:
                        accessible_columns += 1
                
                # Build role text for embedding
                role_text = f"{role_name} {description}".strip()
                if not role_text:
                    role_text = role_name or db_id
                
                role_specs.append({
                    "db_id": db_id,
                    "role": role_name,
                    "description": description,
                    "policy": policy,
                    "accessible_tables": accessible_tables,
                    "accessible_columns": accessible_columns,
                    "total_tables": total_tables,
                    "total_columns": total_columns,
                    "role_text": role_text,
                    "schema_text": schema_text,
                    "is_system_manager": role_name.strip().lower() == "systemmanager"
                })
        
        return role_specs
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def evaluate_semantic_quality(self) -> Dict[str, SemanticMetrics]:
        """
        Evaluate semantic alignment between roles and database schemas.
        
        Uses embedding-based cosine similarity to measure:
        - Role-schema semantic alignment
        - Column coverage ratios
        - Role diversity (std dev of coverage)
        
        Returns:
            Dict mapping db_id to SemanticMetrics
        """
        if not self.role_assignments:
            raise ValueError("Role assignments not loaded.")
        if not self.schema_summaries:
            raise ValueError("Schema summaries not loaded.")
        
        # Compute embeddings
        self._compute_embeddings()
        
        # Prepare role records with similarities
        role_specs = self._prepare_role_records()
        role_records = []
        
        for spec in role_specs:
            schema_emb = self._schema_emb_dict.get(spec["schema_text"])
            role_emb = self._role_emb_dict.get(spec["role_text"])
            
            # Compute similarity (normalized to [0, 1])
            raw_sim = self.cosine_similarity(role_emb, schema_emb)
            similarity = max(0.0, min(1.0, (raw_sim + 1.0) / 2.0))
            
            # Coverage ratios - now properly calculated even for "*" wildcards
            if spec["total_columns"] > 0:
                column_coverage = spec["accessible_columns"] / spec["total_columns"]
            else:
                column_coverage = 0
            
            role_records.append({
                "db_id": spec["db_id"],
                "role": spec["role"],
                "similarity": similarity,
                "column_coverage": column_coverage,
                "is_system_manager": spec["is_system_manager"]
            })
        
        # Aggregate by database
        metrics = {}
        for db_id in sorted(set(r["db_id"] for r in role_records)):
            db_roles = [r for r in role_records if r["db_id"] == db_id]
            # Exclude SystemManager from coverage calculation (expected to be 100%)
            non_sm_roles = [r for r in db_roles if not r["is_system_manager"]]
            
            avg_similarity = np.mean([r["similarity"] for r in db_roles])
            avg_coverage = np.mean([r["column_coverage"] for r in non_sm_roles]) if non_sm_roles else 0
            coverage_std = np.std([r["column_coverage"] for r in non_sm_roles]) if len(non_sm_roles) > 1 else 0
            
            # Only semantic alignment issue here - diversity moved to evaluate_policy_overlap
            issues = []
            if avg_similarity < self.SIMILARITY_MIN:
                issues.append(f"Low semantic alignment ({avg_similarity:.2f} < {self.SIMILARITY_MIN})")
            
            metrics[db_id] = SemanticMetrics(
                db_id=db_id,
                avg_similarity=avg_similarity,
                avg_coverage=avg_coverage,
                coverage_std=coverage_std,
                num_roles=len(db_roles),
                issues=issues
            )
        
        return metrics
    
    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def count_policy_columns(policy: dict, db_schema: dict = None) -> int:
        """Count actual columns in policy, expanding '*' wildcard if schema is provided.
        
        Args:
            policy: Dict of {table: [columns]} or {table: "*"}
            db_schema: Schema dict with format {table: {'columns': [{'name': ...}]}}
        
        Returns:
            Total column count with '*' properly expanded
        """
        total = 0
        for table, cols in policy.items():
            if cols == ["*"] or cols == "*":
                if db_schema:
                    for t_name, t_info in db_schema.items():
                        if t_name.lower() == table.lower():
                            total += len(t_info['columns'])
                            break
                    else:
                        total += 1
                else:
                    total += 1
            elif isinstance(cols, list):
                total += len(cols)
            else:
                total += 1
        return total
    
    @staticmethod
    def has_wildcard_policy(policy: dict) -> bool:
        """Check if policy contains '*' wildcard."""
        return any(c == ["*"] or c == "*" for c in policy.values())
    
    @staticmethod
    def policy_to_column_set(policy: dict) -> set:
        """Convert policy dict to set of table.column strings."""
        cols = set()
        for table, columns in policy.items():
            if isinstance(columns, list):
                for col in columns:
                    cols.add(f"{table}.{col}")
            else:
                cols.add(f"{table}.{columns}")
        return cols
    
    def evaluate_policy_overlap(self) -> Dict[str, OverlapMetrics]:
        """
        Evaluate policy overlap between roles using Jaccard similarity.
        
        High overlap (>70%) indicates redundant or poorly differentiated roles.
        
        Returns:
            Dict mapping db_id to OverlapMetrics
        """
        if not self.role_assignments:
            raise ValueError("Role assignments not loaded.")
        
        metrics = {}
        
        for db_id in sorted(self.role_assignments.keys()):
            roles = self.role_assignments[db_id]
            
            # Filter non-SystemManager roles
            non_sm_roles = [
                r for r in roles 
                if r.get("role", "").strip().lower() != "systemmanager"
            ]
            
            if len(non_sm_roles) < 2:
                continue
            
            # Calculate pairwise Jaccard similarities
            overlaps = []
            for i, role1 in enumerate(non_sm_roles):
                set1 = self.policy_to_column_set(role1.get("policy", {}))
                for j, role2 in enumerate(non_sm_roles):
                    if i >= j:
                        continue
                    set2 = self.policy_to_column_set(role2.get("policy", {}))
                    overlap = self.jaccard_similarity(set1, set2)
                    overlaps.append({
                        "role1": role1.get("role"),
                        "role2": role2.get("role"),
                        "overlap": overlap
                    })
            
            if overlaps:
                avg_overlap = np.mean([o["overlap"] for o in overlaps])
                max_overlap = max(o["overlap"] for o in overlaps)
                max_pair = max(overlaps, key=lambda x: x["overlap"])
                
                # Calculate coverage diversity (moved from semantic eval - no embedding needed)
                # Coverage = fraction of total columns a role can access
                schema_summary = self.schema_summaries.get(db_id, {}) if self.schema_summaries else {}
                total_columns = schema_summary.get("column_count", 0)
                table_columns = schema_summary.get("table_columns", {})
                
                coverages = []
                for role in non_sm_roles:
                    policy = role.get("policy", {})
                    accessible = 0
                    for table, cols in policy.items():
                        if cols == ["*"] or cols == "*":
                            # Find actual column count for this table (case-insensitive)
                            for t_name, t_cols in table_columns.items():
                                if t_name.lower() == table.lower():
                                    accessible += t_cols
                                    break
                        elif isinstance(cols, list):
                            accessible += len(cols)
                    if total_columns > 0:
                        coverages.append(accessible / total_columns)
                
                avg_coverage = np.mean(coverages) if coverages else 0
                coverage_std = np.std(coverages) if len(coverages) > 1 else 0
                
                # Issues: overlap, diversity, and coverage
                issues = []
                if max_overlap > self.OVERLAP_MAX:
                    issues.append(f"High max overlap ({max_overlap:.0%} > {self.OVERLAP_MAX:.0%}) between {max_pair['role1']} and {max_pair['role2']} - consider differentiating these roles")
                if coverage_std < self.COVERAGE_STD_MIN and len(coverages) > 1:
                    issues.append(f"Low role diversity (coverage_std={coverage_std:.3f} < {self.COVERAGE_STD_MIN})")
                if avg_coverage > self.COVERAGE_MAX and len(coverages) > 0:
                    issues.append(f"Most roles too permissive (coverage {avg_coverage:.0%} > {self.COVERAGE_MAX:.0%})")
                
                metrics[db_id] = OverlapMetrics(
                    db_id=db_id,
                    avg_overlap=avg_overlap,
                    max_overlap=max_overlap,
                    max_pair=(max_pair["role1"], max_pair["role2"]),
                    num_pairs=len(overlaps),
                    issues=issues
                )
        
        return metrics
    
    def evaluate_all(self) -> QualityReport:
        """
        Run all quality evaluations and generate comprehensive report.
        
        Returns:
            QualityReport with all metrics and issue summary
        """
        report = QualityReport()
        
        # Run evaluations
        if self.dataset:
            report.deny_rate_metrics = self.evaluate_deny_rates()
            report.total_entries = len(self.dataset)
        
        if self.role_assignments and self.schema_summaries:
            report.semantic_metrics = self.evaluate_semantic_quality()
        
        if self.role_assignments:
            report.overlap_metrics = self.evaluate_policy_overlap()
            
            # Count totals
            report.total_databases = len(self.role_assignments)
            report.total_roles = sum(len(roles) for roles in self.role_assignments.values())
        
        # Aggregate all issues
        all_db_ids = set()
        if report.deny_rate_metrics:
            all_db_ids.update(report.deny_rate_metrics.keys())
        if report.semantic_metrics:
            all_db_ids.update(report.semantic_metrics.keys())
        if report.overlap_metrics:
            all_db_ids.update(report.overlap_metrics.keys())
        
        for db_id in sorted(all_db_ids):
            issues = []
            
            if db_id in report.deny_rate_metrics and report.deny_rate_metrics[db_id].issues:
                issues.extend(report.deny_rate_metrics[db_id].issues)
            if db_id in report.semantic_metrics and report.semantic_metrics[db_id].issues:
                issues.extend(report.semantic_metrics[db_id].issues)
            if db_id in report.overlap_metrics and report.overlap_metrics[db_id].issues:
                issues.extend(report.overlap_metrics[db_id].issues)
            
            if issues:
                report.all_issues[db_id] = issues
                report.databases_with_issues.append(db_id)
        
        return report


def print_quality_report(report: QualityReport, verbose: bool = True):
    """
    Print formatted quality report.
    
    Args:
        report: QualityReport from evaluator
        verbose: Whether to print detailed per-database info
    """
    print("=" * 80)
    print("ROLE GENERATION QUALITY REPORT")
    print("=" * 80)
    
    # Summary statistics
    print(f"\nOverall Statistics:")
    print(f"   Total databases: {report.total_databases}")
    print(f"   Total roles: {report.total_roles}")
    print(f"   Total entries: {report.total_entries}")
    print(f"   Databases with issues: {len(report.databases_with_issues)}")
    print(f"   Pass rate: {report.pass_rate:.1%}")
    
    # Deny rate summary
    if report.deny_rate_metrics and verbose:
        print("\n" + "-" * 80)
        print("Deny Rate Analysis")
        print("-" * 80)
        print(f"{'Database':<30} {'Deny%':>8} {'All-Allow%':>12} {'Status':>10}")
        print("-" * 80)
        
        for db_id, m in sorted(report.deny_rate_metrics.items()):
            status = "[WARN]" if m.issues else "[OK]"
            print(f"{db_id:<30} {m.deny_rate:>7.1f}% {m.all_allow_pct:>11.1f}% {status:>10}")
    
    # Semantic quality summary
    if report.semantic_metrics and verbose:
        print("\n" + "-" * 80)
        print("Semantic Quality Analysis")
        print("-" * 80)
        print(f"{'Database':<30} {'Similarity':>12} {'Coverage':>10} {'Diversity':>12} {'Status':>10}")
        print("-" * 80)
        
        for db_id, m in sorted(report.semantic_metrics.items()):
            status = "[WARN]" if m.issues else "[OK]"
            print(f"{db_id:<30} {m.avg_similarity:>11.4f} {m.avg_coverage:>9.1%} {m.coverage_std:>11.4f} {status:>10}")
    
    # Policy overlap summary
    if report.overlap_metrics and verbose:
        print("\n" + "-" * 80)
        print("Policy Overlap Analysis")
        print("-" * 80)
        print(f"{'Database':<30} {'Avg Overlap':>12} {'Max Overlap':>12} {'Status':>10}")
        print("-" * 80)
        
        for db_id, m in sorted(report.overlap_metrics.items()):
            status = "[WARN]" if m.issues else "[OK]"
            print(f"{db_id:<30} {m.avg_overlap:>11.1%} {m.max_overlap:>11.1%} {status:>10}")
    
    # Issues summary
    print("\n" + "=" * 80)
    print("Issues Summary")
    print("=" * 80)
    
    if report.all_issues:
        for db_id, issues in report.all_issues.items():
            print(f"\n[WARN] {db_id}:")
            for issue in issues:
                print(f"   - {issue}")
    else:
        print("\n[OK] No quality issues detected!")
    
    print("\n" + "=" * 80)


# ============================================================================
# CRUD-Level Role Quality Evaluator
# ============================================================================

class CrudRoleQualityEvaluator:
    """
    Quality evaluator for CRUD-level RBAC datasets (LiveSQLBench-Full).
    
    Unlike Column-Level RBAC which focuses on SELECT permissions for specific columns,
    CRUD-Level RBAC evaluates DDL, INSERT, DELETE, and UPDATE permissions.
    
    Quality Thresholds:
        - DENY_RATE_MIN: 5% - Minimum acceptable deny rate
        - DENY_RATE_MAX: 90% - Maximum acceptable deny rate
        - OPERATION_BALANCE_MIN: 10% - Min distribution per operation type
        - OVERLAP_MAX: 70% - Maximum acceptable policy overlap
        - COVERAGE_STD_MIN: 0.1 - Minimum coverage std dev for diversity
    
    Usage:
        evaluator = CrudRoleQualityEvaluator(
            role_assignments_path="outputs/crud_role_assignments_livesqlbench.json",
            dataset_path="outputs/crud_rbac_dataset_livesqlbench.json"
        )
        report = evaluator.evaluate_all()
    """
    
    # Quality thresholds
    DENY_RATE_MIN = 5.0    # Minimum acceptable deny rate (%)
    DENY_RATE_MAX = 90.0   # Maximum acceptable deny rate (%)
    OPERATION_BALANCE_MIN = 5.0  # Min distribution per operation type (%)
    OVERLAP_MAX = 0.7      # Maximum acceptable policy overlap (Jaccard)
    COVERAGE_STD_MIN = 0.1 # Minimum coverage std dev for diversity
    
    # CRUD operation keywords for SQL classification
    DDL_KEYWORDS = {'CREATE', 'DROP', 'ALTER', 'TRUNCATE'}
    INSERT_KEYWORDS = {'INSERT'}
    DELETE_KEYWORDS = {'DELETE'}
    UPDATE_KEYWORDS = {'UPDATE'}
    SELECT_KEYWORDS = {'SELECT'}
    
    def __init__(
        self,
        role_assignments_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        role_assignments: Optional[Dict] = None,
        dataset: Optional[List[Dict]] = None,
        schema_summaries: Optional[Dict[str, str]] = None,
        schemas: Optional[Dict[str, Dict[str, List[str]]]] = None,
        embed_model: str = "text-embedding-3-small",
        embed_backend: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize CRUD quality evaluator.
        
        Args:
            role_assignments_path: Path to role assignments JSON file
            dataset_path: Path to RBAC dataset JSON file
            role_assignments: Pre-loaded role assignments dict
            dataset: Pre-loaded dataset list
            schema_summaries: Dict mapping db_id to schema text (for semantic evaluation)
            schemas: Dict mapping db_id to {table_name: [column_names]} (for coverage calculation)
            embed_model: Embedding model name (default: text-embedding-3-small)
                - OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
                - BAAI/fastembed: "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"
            embed_backend: 'openai' or 'fastembed' (auto-detected if None)
            thresholds: Dict of quality thresholds to override defaults:
                - deny_rate_min: Minimum deny rate (default: 5.0)
                - deny_rate_max: Maximum deny rate (default: 90.0)
                - coverage_std_min: Minimum coverage std (default: 0.1)
                - max_overlap: Maximum policy overlap (default: 0.7)
                - similarity_min: Minimum semantic similarity (default: 0.6)
        """
        self.role_assignments = role_assignments
        self.dataset = dataset
        self.schema_summaries = schema_summaries or {}
        self.schemas = schemas or {}
        self.embed_model = embed_model
        self.embed_backend = embed_backend
        
        # Embedding cache for semantic evaluation
        self._schema_emb_dict: Dict[str, np.ndarray] = {}
        self._role_emb_dict: Dict[str, np.ndarray] = {}
        
        # Semantic threshold
        self.SIMILARITY_MIN = 0.6
        
        # Override thresholds if provided
        if thresholds:
            if "deny_rate_min" in thresholds:
                self.DENY_RATE_MIN = thresholds["deny_rate_min"]
            if "deny_rate_max" in thresholds:
                self.DENY_RATE_MAX = thresholds["deny_rate_max"]
            if "coverage_std_min" in thresholds:
                self.COVERAGE_STD_MIN = thresholds["coverage_std_min"]
            if "max_overlap" in thresholds:
                self.OVERLAP_MAX = thresholds["max_overlap"]
            if "similarity_min" in thresholds:
                self.SIMILARITY_MIN = thresholds["similarity_min"]
        
        # Load from files if paths provided
        if role_assignments_path:
            with open(role_assignments_path, 'r') as f:
                self.role_assignments = json.load(f)
        
        if dataset_path:
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)
    
    # Operation type mapping from raw values to standard CRUD categories
    OPERATION_MAP = {
        # DDL operations
        'CREATE': 'DDL', 'DROP': 'DDL', 'ALTER': 'DDL', 'TRUNCATE': 'DDL',
        'ANALYZE': 'DDL',  # PostgreSQL ANALYZE
        # DML operations
        'SELECT': 'SELECT', 'UNION': 'SELECT',  # UNION is a SELECT variant
        'INSERT': 'INSERT',
        'UPDATE': 'UPDATE',
        'DELETE': 'DELETE',
        # Standard categories map to themselves
        'DDL': 'DDL',
    }
    
    def _get_entry_fields(self, entry: Dict) -> Dict:
        """
        Extract fields from dataset entry, supporting both:
        1. LiveSQLBench format (flat structure with 'allowed', 'gold_sql', 'operation' at top level)
        2. Spider/Bird format (nested 'metadata' dict with 'permission', 'gold_sql')
        
        Returns:
            Dict with normalized fields: permission, gold_sql, operation
        """
        # Try LiveSQLBench format first (flat structure)
        if 'allowed' in entry:
            # LiveSQLBench: 'allowed' is bool, 'gold_sql' and 'operation' at top level
            allowed = entry.get('allowed', False)
            permission = 'allowed' if allowed else 'denied'
            gold_sql = entry.get('gold_sql', '')
            # LiveSQLBench has pre-classified 'operation' field - map to standard categories
            raw_operation = entry.get('operation', '').upper()
            operation = self.OPERATION_MAP.get(raw_operation)
            if not operation:
                # Fallback to SQL-based classification
                operation = self.classify_sql_operation(gold_sql)
            return {
                'permission': permission,
                'gold_sql': gold_sql,
                'operation': operation
            }
        
        # Spider/Bird format: nested metadata dict
        metadata = entry.get('metadata', {})
        permission = metadata.get('permission', 'denied')
        gold_sql = metadata.get('gold_sql', '')
        operation = self.classify_sql_operation(gold_sql)
        return {
            'permission': permission,
            'gold_sql': gold_sql,
            'operation': operation
        }
    
    @staticmethod
    def classify_sql_operation(sql: str) -> str:
        """
        Classify SQL statement into operation type.
        
        Args:
            sql: SQL statement to classify
            
        Returns:
            Operation type: 'DDL', 'INSERT', 'DELETE', 'UPDATE', 'SELECT', or 'UNKNOWN'
        """
        if not sql:
            return 'UNKNOWN'
        
        # Normalize SQL - remove comments and leading whitespace
        sql_upper = sql.strip().upper()
        
        # Handle statements starting with WITH (CTEs)
        if sql_upper.startswith('WITH'):
            # Find the main operation after WITH clause
            # Look for SELECT, INSERT, UPDATE, DELETE after WITH
            for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']:
                if keyword in sql_upper:
                    if keyword == 'SELECT':
                        return 'SELECT'
                    elif keyword == 'INSERT':
                        return 'INSERT'
                    elif keyword == 'UPDATE':
                        return 'UPDATE'
                    elif keyword == 'DELETE':
                        return 'DELETE'
        
        # Check for DDL operations
        for kw in CrudRoleQualityEvaluator.DDL_KEYWORDS:
            if sql_upper.startswith(kw):
                return 'DDL'
        
        # Check for DML operations
        if sql_upper.startswith('INSERT'):
            return 'INSERT'
        if sql_upper.startswith('DELETE'):
            return 'DELETE'
        if sql_upper.startswith('UPDATE'):
            return 'UPDATE'
        if sql_upper.startswith('SELECT'):
            return 'SELECT'
        
        # Handle DO $$ blocks (PostgreSQL procedural code)
        if sql_upper.startswith('DO'):
            return 'DDL'  # Treat procedural blocks as DDL
        
        return 'UNKNOWN'
    
    def evaluate_deny_rates(self) -> Dict[str, CrudDenyRateMetrics]:
        """
        Evaluate deny rate distribution across databases for CRUD operations.
        
        Checks:
        - Overall deny rate (should be 5%-90%)
        - Per-operation deny rates (DDL, INSERT, DELETE, UPDATE)
        - All-allow query percentage
        - All-deny queries (should be 0 - SystemManager coverage)
        
        Returns:
            Dict mapping db_id to CrudDenyRateMetrics
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Provide dataset_path or dataset.")
        
        metrics = {}
        
        # Group entries by database
        db_stats = defaultdict(lambda: {
            'allowed': 0, 'denied': 0, 'roles': set(),
            'ddl_allowed': 0, 'ddl_denied': 0,
            'insert_allowed': 0, 'insert_denied': 0,
            'delete_allowed': 0, 'delete_denied': 0,
            'update_allowed': 0, 'update_denied': 0,
            'select_allowed': 0, 'select_denied': 0,
        })
        
        for entry in self.dataset:
            db_id = entry.get('db_id', 'unknown')
            fields = self._get_entry_fields(entry)
            permission = fields['permission']
            op_type = fields['operation']
            
            db_stats[db_id][permission] += 1
            db_stats[db_id]['roles'].add(entry.get('role', ''))
            
            # Track per-operation stats
            op_key = op_type.lower()
            if op_type != 'UNKNOWN':
                db_stats[db_id][f'{op_key}_{permission}'] += 1
        
        # Analyze query-level patterns
        query_groups = defaultdict(list)
        for entry in self.dataset:
            # Use question + gold_sql as query identifier
            question = entry.get('input', '')[:200]
            fields = self._get_entry_fields(entry)
            gold_sql = fields['gold_sql']
            key = (entry.get('db_id', ''), question, gold_sql)
            query_groups[key].append(entry)
        
        # Count all-allow and all-deny queries per database
        all_allow_by_db = defaultdict(int)
        all_deny_by_db = defaultdict(int)
        query_count_by_db = defaultdict(int)
        
        for key, entries in query_groups.items():
            db_id = key[0]
            query_count_by_db[db_id] += 1
            permissions = [self._get_entry_fields(e)['permission'] for e in entries]
            if all(p == 'allowed' for p in permissions):
                all_allow_by_db[db_id] += 1
            if all(p == 'denied' for p in permissions):
                all_deny_by_db[db_id] += 1
        
        # Build metrics for each database
        for db_id in sorted(db_stats.keys()):
            stats = db_stats[db_id]
            total = stats['allowed'] + stats['denied']
            deny_rate = stats['denied'] / total * 100 if total > 0 else 0
            
            total_queries = query_count_by_db[db_id]
            all_allow_queries = all_allow_by_db[db_id]
            all_deny_queries = all_deny_by_db[db_id]
            all_allow_pct = all_allow_queries / total_queries * 100 if total_queries > 0 else 0
            all_deny_pct = all_deny_queries / total_queries * 100 if total_queries > 0 else 0
            
            issues = []
            if deny_rate < self.DENY_RATE_MIN:
                issues.append(f"deny rate too low ({deny_rate:.1f}% < {self.DENY_RATE_MIN}%)")
            if deny_rate > self.DENY_RATE_MAX:
                issues.append(f"deny rate too high ({deny_rate:.1f}% > {self.DENY_RATE_MAX}%)")
            if all_deny_queries > 0:
                issues.append(f"all-deny queries found ({all_deny_queries}) - SystemManager coverage broken")
            
            metrics[db_id] = CrudDenyRateMetrics(
                db_id=db_id,
                total_entries=total,
                allowed_count=stats['allowed'],
                denied_count=stats['denied'],
                deny_rate=deny_rate,
                ddl_allowed=stats['ddl_allowed'],
                ddl_denied=stats['ddl_denied'],
                insert_allowed=stats['insert_allowed'],
                insert_denied=stats['insert_denied'],
                delete_allowed=stats['delete_allowed'],
                delete_denied=stats['delete_denied'],
                update_allowed=stats['update_allowed'],
                update_denied=stats['update_denied'],
                select_allowed=stats['select_allowed'],
                select_denied=stats['select_denied'],
                all_allow_queries=all_allow_queries,
                all_deny_queries=all_deny_queries,
                total_queries=total_queries,
                all_allow_pct=all_allow_pct,
                all_deny_pct=all_deny_pct,
                roles=list(stats['roles']),
                issues=issues
            )
        
        return metrics
    
    def evaluate_coverage(self) -> Dict[str, CrudCoverageMetrics]:
        """
        Evaluate permission coverage across tables for CRUD operations.
        
        Measures:
        - Tables accessible via INSERT/DELETE/UPDATE by any role
        - Role diversity (tables per role standard deviation)
        
        Returns:
            Dict mapping db_id to CrudCoverageMetrics
        """
        if not self.role_assignments:
            raise ValueError("Role assignments not loaded.")
        
        metrics = {}
        
        for db_id, roles_list in sorted(self.role_assignments.items()):
            # Get schema info (ground truth)
            base_db_id = db_id[:-2] if db_id.endswith('_M') else db_id
            schema_tables = self.schemas.get(db_id, self.schemas.get(base_db_id, {}))
            
            # Calculate schema totals
            total_schema_tables = len(schema_tables) if schema_tables else 0
            total_schema_columns = sum(len(cols) for cols in schema_tables.values()) if schema_tables else 0
            
            # Build schema table set (lowercase)
            schema_table_set = {t.lower() for t in schema_tables.keys()}
            
            # Track per-role coverages
            insert_coverages = []  # Per-role INSERT coverage (table-level)
            delete_coverages = []  # Per-role DELETE coverage (table-level)
            select_coverages = []  # Per-role SELECT coverage (column-level)
            update_coverages = []  # Per-role UPDATE coverage (column-level)
            
            has_ddl_role = False
            permissions_per_role = []  # For diversity calculation
            
            for role in roles_list:
                role_name = role.get('role', '')
                is_system_manager = role_name.strip().lower() == 'systemmanager'
                
                # DDL permission (database-level)
                if role.get('DDL', False):
                    has_ddl_role = True
                
                # Skip SystemManager for coverage calculation
                if is_system_manager:
                    continue
                
                # INSERT coverage (table-level)
                insert_tables = role.get('INSERT', [])
                if isinstance(insert_tables, list):
                    role_insert_count = len([t for t in insert_tables if t.lower() in schema_table_set])
                    insert_cov = role_insert_count / total_schema_tables if total_schema_tables > 0 else 0
                else:
                    insert_cov = 0
                insert_coverages.append(insert_cov)
                
                # DELETE coverage (table-level)
                delete_tables = role.get('DELETE', [])
                if isinstance(delete_tables, list):
                    role_delete_count = len([t for t in delete_tables if t.lower() in schema_table_set])
                    delete_cov = role_delete_count / total_schema_tables if total_schema_tables > 0 else 0
                else:
                    delete_cov = 0
                delete_coverages.append(delete_cov)
                
                # SELECT/UPDATE coverage (column-level)
                tables_dict = role.get('tables', {})
                role_select_cols = 0
                role_update_cols = 0
                
                if isinstance(tables_dict, dict):
                    for table, perms in tables_dict.items():
                        table_lower = table.lower()
                        if isinstance(perms, dict):
                            # SELECT columns
                            select_cols = perms.get('SELECT', [])
                            if select_cols == ['*']:
                                # Wildcard: count all columns from schema for this table
                                for schema_table, cols in schema_tables.items():
                                    if schema_table.lower() == table_lower:
                                        role_select_cols += len(cols)
                            elif isinstance(select_cols, list):
                                role_select_cols += len(select_cols)
                            
                            # UPDATE columns
                            update_cols = perms.get('UPDATE', [])
                            if update_cols == ['*']:
                                # Wildcard: count all columns from schema for this table
                                for schema_table, cols in schema_tables.items():
                                    if schema_table.lower() == table_lower:
                                        role_update_cols += len(cols)
                            elif isinstance(update_cols, list):
                                role_update_cols += len(update_cols)
                
                select_cov = role_select_cols / total_schema_columns if total_schema_columns > 0 else 0
                update_cov = role_update_cols / total_schema_columns if total_schema_columns > 0 else 0
                select_coverages.append(select_cov)
                update_coverages.append(update_cov)
                
                # Build permission set for diversity calculation
                # IMPORTANT: All operations are normalized to column-level for fair comparison
                # INSERT/DELETE on a table = permissions on ALL columns of that table
                role_perm_set = set()
                
                # INSERT: expand to column-level (table INSERT = all columns INSERT)
                for t in (insert_tables if isinstance(insert_tables, list) else []):
                    t_lower = t.lower()
                    # Find columns for this table from schema
                    for schema_table, cols in schema_tables.items():
                        if schema_table.lower() == t_lower:
                            for col in cols:
                                role_perm_set.add(('INSERT', t_lower, col.lower()))
                            break
                    else:
                        # Table not in schema, add as single permission
                        role_perm_set.add(('INSERT', t_lower, '*'))
                
                # DELETE: expand to column-level (table DELETE = all columns DELETE)
                for t in (delete_tables if isinstance(delete_tables, list) else []):
                    t_lower = t.lower()
                    for schema_table, cols in schema_tables.items():
                        if schema_table.lower() == t_lower:
                            for col in cols:
                                role_perm_set.add(('DELETE', t_lower, col.lower()))
                            break
                    else:
                        role_perm_set.add(('DELETE', t_lower, '*'))
                
                # SELECT/UPDATE: already column-level, expand ['*'] to actual columns
                if isinstance(tables_dict, dict):
                    for table, perms in tables_dict.items():
                        table_lower = table.lower()
                        if isinstance(perms, dict):
                            # SELECT
                            select_cols = perms.get('SELECT', [])
                            if select_cols == ['*']:
                                for schema_table, cols in schema_tables.items():
                                    if schema_table.lower() == table_lower:
                                        for col in cols:
                                            role_perm_set.add(('SELECT', table_lower, col.lower()))
                            else:
                                for col in select_cols:
                                    role_perm_set.add(('SELECT', table_lower, col.lower()))
                            # UPDATE
                            update_cols = perms.get('UPDATE', [])
                            if update_cols == ['*']:
                                for schema_table, cols in schema_tables.items():
                                    if schema_table.lower() == table_lower:
                                        for col in cols:
                                            role_perm_set.add(('UPDATE', table_lower, col.lower()))
                            else:
                                for col in update_cols:
                                    role_perm_set.add(('UPDATE', table_lower, col.lower()))
                
                permissions_per_role.append(role_perm_set)
            
            # Calculate average coverages
            avg_insert_cov = np.mean(insert_coverages) if insert_coverages else 0
            avg_delete_cov = np.mean(delete_coverages) if delete_coverages else 0
            avg_select_cov = np.mean(select_coverages) if select_coverages else 0
            avg_update_cov = np.mean(update_coverages) if update_coverages else 0
            
            # ============================================================
            # Per-operation coverage std (NEW: separate std for each op)
            # ============================================================
            select_std = np.std(select_coverages) if len(select_coverages) > 1 else 0
            update_std = np.std(update_coverages) if len(update_coverages) > 1 else 0
            insert_std = np.std(insert_coverages) if len(insert_coverages) > 1 else 0
            delete_std = np.std(delete_coverages) if len(delete_coverages) > 1 else 0
            
            # Data-driven weights from livesqlbench-full dataset operation distribution:
            # SELECT: 140/184 = 76.09%, UPDATE: 24/184 = 13.04%, 
            # INSERT: 14/184 = 7.61%, DELETE: 6/184 = 3.26%
            weighted_std = (
                0.7609 * select_std + 
                0.1304 * update_std + 
                0.0761 * insert_std + 
                0.0326 * delete_std
            )
            
            num_roles = len(roles_list)
            avg_perms = np.mean([len(ps) for ps in permissions_per_role]) if permissions_per_role else 0
            
            issues = []
            # Check SELECT std as primary diversity indicator
            if select_std < self.COVERAGE_STD_MIN and len(select_coverages) > 1:
                issues.append(f"Low SELECT diversity (std={select_std:.3f} < {self.COVERAGE_STD_MIN})")
            
            metrics[db_id] = CrudCoverageMetrics(
                db_id=db_id,
                total_tables=total_schema_tables,
                total_columns=total_schema_columns,
                avg_insert_coverage=avg_insert_cov,
                avg_delete_coverage=avg_delete_cov,
                avg_select_coverage=avg_select_cov,
                avg_update_coverage=avg_update_cov,
                has_ddl_role=has_ddl_role,
                num_roles=num_roles,
                avg_permissions_per_role=avg_perms,
                select_coverage_std=select_std,
                update_coverage_std=update_std,
                insert_coverage_std=insert_std,
                delete_coverage_std=delete_std,
                coverage_std=weighted_std,  # Backward compatible weighted std
                issues=issues
            )
        
        return metrics
    
    @staticmethod
    def count_crud_permissions(role_def: dict, schema: dict = None) -> dict:
        """
        Count permissions in a CRUD role definition.
        When schema is provided, expand ['*'] to actual column count.
        
        Args:
            role_def: CRUD role definition
            schema: Optional schema dict {table: [columns]} to expand wildcards
            
        Returns:
            Dict with ddl, insert, delete, select_cols, update_cols counts
        """
        select_cols = 0
        update_cols = 0
        
        tables_config = role_def.get('tables', {})
        for table, perms in tables_config.items():
            # Get actual column count from schema if available
            schema_col_count = 0
            if schema:
                t_lower = table.lower()
                for schema_table, cols in schema.items():
                    if schema_table.lower() == t_lower:
                        schema_col_count = len(cols)
                        break
            
            # SELECT columns
            sel = perms.get('SELECT', [])
            if sel == ['*'] and schema_col_count > 0:
                select_cols += schema_col_count
            elif isinstance(sel, list):
                select_cols += len(sel)
            
            # UPDATE columns
            upd = perms.get('UPDATE', [])
            if upd == ['*'] and schema_col_count > 0:
                update_cols += schema_col_count
            elif isinstance(upd, list):
                update_cols += len(upd)
        
        return {
            'ddl': 1 if role_def.get('DDL', False) else 0,
            'insert': len(role_def.get('INSERT', [])),
            'delete': len(role_def.get('DELETE', [])),
            'select_cols': select_cols,
            'update_cols': update_cols,
        }
    
    @staticmethod
    def calculate_select_coverage(role_def: dict, schema: dict) -> float:
        """
        Calculate SELECT coverage ratio for a single CRUD role.
        
        SELECT is the primary permission differentiator in CRUD-level RBAC.
        Coverage = SELECT columns / total schema columns
        
        Args:
            role_def: CRUD role definition
            schema: Schema dict {table: [columns]}
            
        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if not schema:
            return 0.0
        
        total_schema_cols = sum(len(cols) for cols in schema.values())
        if total_schema_cols == 0:
            return 0.0
        
        select_cols = set()
        tables_config = role_def.get('tables', {})
        
        for t, perms in tables_config.items():
            t_lower = t.lower()
            # Find matching schema table
            schema_cols = []
            for schema_table, cols in schema.items():
                if schema_table.lower() == t_lower:
                    schema_cols = cols
                    break
            
            cols = perms.get('SELECT', [])
            if cols == ['*']:
                for col in schema_cols:
                    select_cols.add((t_lower, col.lower()))
            elif isinstance(cols, list):
                for col in cols:
                    select_cols.add((t_lower, col.lower()))
        
        return len(select_cols) / total_schema_cols
    
    @staticmethod
    def calculate_select_overlap(role1: dict, role2: dict, schema: dict) -> float:
        """
        Calculate Jaccard similarity between two CRUD roles based on SELECT permissions.
        
        Args:
            role1, role2: CRUD role definitions
            schema: Schema dict {table: [columns]} for wildcard expansion
            
        Returns:
            Jaccard similarity (0.0 to 1.0)
        """
        def to_select_set(role_def):
            select_cols = set()
            tables_config = role_def.get('tables', {})
            
            for t, perms in tables_config.items():
                t_lower = t.lower()
                schema_cols = []
                for schema_table, cols in schema.items():
                    if schema_table.lower() == t_lower:
                        schema_cols = cols
                        break
                
                cols = perms.get('SELECT', [])
                if cols == ['*']:
                    for col in schema_cols:
                        select_cols.add((t_lower, col.lower()))
                elif isinstance(cols, list):
                    for col in cols:
                        select_cols.add((t_lower, col.lower()))
            return select_cols
        
        set1 = to_select_set(role1)
        set2 = to_select_set(role2)
        
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def has_wildcard_crud_policy(role_def: dict) -> bool:
        """
        Check if CRUD role has wildcard/full administrative access.
        
        Only DDL=True grants full admin access. SELECT/UPDATE with ['*']
        just means access to all columns of specific tables, not admin access.
        
        Args:
            role_def: CRUD role definition
            
        Returns:
            True if role has DDL (full admin) access
        """
        # Only DDL=True means full administrative access
        return role_def.get('DDL', False) == True
    
    @staticmethod
    def crud_policy_to_permission_set(role_def: dict, schema: dict = None) -> set:
        """
        Convert CRUD role definition to set of permission tuples.
        
        Args:
            role_def: CRUD role definition with DDL, INSERT, DELETE, tables structure
            schema: Optional schema dict {table: [columns]} to expand ['*'] wildcards
            
        Returns:
            Set of permission tuples like ('INSERT', 'users'), ('SELECT', 'orders', 'id')
        """
        perms = set()
        
        # DDL permission (database-level)
        if role_def.get('DDL', False):
            perms.add(('DDL', '*'))
        
        # INSERT permission (table-level)
        insert_tables = role_def.get('INSERT', [])
        if isinstance(insert_tables, list):
            for table in insert_tables:
                perms.add(('INSERT', table.lower()))
        
        # DELETE permission (table-level)
        delete_tables = role_def.get('DELETE', [])
        if isinstance(delete_tables, list):
            for table in delete_tables:
                perms.add(('DELETE', table.lower()))
        
        # SELECT/UPDATE permissions (column-level in 'tables' dict)
        tables_dict = role_def.get('tables', {})
        if isinstance(tables_dict, dict):
            for table, table_perms in tables_dict.items():
                table_lower = table.lower()
                if isinstance(table_perms, dict):
                    # Get schema columns for this table (for wildcard expansion)
                    schema_cols = []
                    if schema:
                        for schema_table, cols in schema.items():
                            if schema_table.lower() == table_lower:
                                schema_cols = cols
                                break
                    
                    # SELECT columns
                    select_cols = table_perms.get('SELECT', [])
                    if select_cols == ['*'] and schema_cols:
                        # Expand wildcard to actual columns
                        for col in schema_cols:
                            perms.add(('SELECT', table_lower, col.lower()))
                    elif isinstance(select_cols, list):
                        for col in select_cols:
                            perms.add(('SELECT', table_lower, col.lower()))
                    
                    # UPDATE columns
                    update_cols = table_perms.get('UPDATE', [])
                    if update_cols == ['*'] and schema_cols:
                        # Expand wildcard to actual columns
                        for col in schema_cols:
                            perms.add(('UPDATE', table_lower, col.lower()))
                    elif isinstance(update_cols, list):
                        for col in update_cols:
                            perms.add(('UPDATE', table_lower, col.lower()))
        
        return perms

    def evaluate_policy_overlap(self) -> Dict[str, OverlapMetrics]:
        """
        Evaluate policy overlap between roles using Jaccard similarity.
        
        High overlap (>70%) indicates redundant or poorly differentiated roles.
        Now expands ['*'] wildcards using schema for accurate column-level comparison.
        
        Returns:
            Dict mapping db_id to OverlapMetrics
        """
        if not self.role_assignments:
            raise ValueError("Role assignments not loaded.")
        
        metrics = {}
        
        for db_id in sorted(self.role_assignments.keys()):
            roles = self.role_assignments[db_id]
            
            # Get schema for this database (for wildcard expansion)
            base_db_id = db_id[:-2] if db_id.endswith('_M') else db_id
            schema = self.schemas.get(db_id, self.schemas.get(base_db_id, {}))
            
            # Filter non-SystemManager roles (also exclude Admin variants)
            non_sm_roles = [
                r for r in roles 
                if r.get("role", "").strip().lower() != "systemmanager"
                and 'admin' not in r.get("role", "").lower()
            ]
            
            if len(non_sm_roles) < 2:
                continue
            
            # Calculate pairwise Jaccard similarities (with schema for wildcard expansion)
            overlaps = []
            for i, role1 in enumerate(non_sm_roles):
                set1 = self.crud_policy_to_permission_set(role1, schema)
                for j, role2 in enumerate(non_sm_roles):
                    if i >= j:
                        continue
                    set2 = self.crud_policy_to_permission_set(role2, schema)
                    
                    # Jaccard similarity
                    if not set1 and not set2:
                        overlap = 1.0
                    else:
                        intersection = len(set1 & set2)
                        union = len(set1 | set2)
                        overlap = intersection / union if union > 0 else 0.0
                    
                    overlaps.append({
                        "role1": role1.get("role"),
                        "role2": role2.get("role"),
                        "overlap": overlap
                    })
            
            if overlaps:
                avg_overlap = np.mean([o["overlap"] for o in overlaps])
                max_overlap = max(o["overlap"] for o in overlaps)
                max_pair = max(overlaps, key=lambda x: x["overlap"])
                
                issues = []
                if max_overlap > self.OVERLAP_MAX:
                    issues.append(
                        f"High max overlap ({max_overlap:.0%} > {self.OVERLAP_MAX:.0%}) "
                        f"between {max_pair['role1']} and {max_pair['role2']}"
                    )
                
                metrics[db_id] = OverlapMetrics(
                    db_id=db_id,
                    avg_overlap=avg_overlap,
                    max_overlap=max_overlap,
                    max_pair=(max_pair["role1"], max_pair["role2"]),
                    num_pairs=len(overlaps),
                    issues=issues
                )
        
        return metrics
    
    def get_operation_distribution(self) -> Dict[str, int]:
        """
        Get distribution of SQL operation types in the dataset.
        
        Returns:
            Dict mapping operation type to count
        """
        if not self.dataset:
            return {}
        
        distribution = Counter()
        for entry in self.dataset:
            fields = self._get_entry_fields(entry)
            op_type = fields['operation']
            distribution[op_type] += 1
        
        return dict(distribution)
    
    @staticmethod
    def cosine_similarity(vec1: Optional[np.ndarray], vec2: Optional[np.ndarray]) -> float:
        """Compute cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _format_role_text(self, role: Dict) -> str:
        """Format role definition to text for embedding.
        
        For semantic similarity, we only use role name and description,
        NOT the table/column names from permissions. This measures true
        semantic alignment between role's business meaning and schema content,
        not just lexical overlap from table names.
        """
        parts = []
        
        role_name = role.get('role', 'Unknown')
        parts.append(f"Role: {role_name}")
        
        description = role.get('description', '')
        if description:
            parts.append(f"Description: {description}")
        
        # Note: We intentionally exclude table/column names from permissions
        # to avoid artificial similarity inflation from lexical overlap
        
        return ". ".join(parts)
    
    def evaluate_semantic_quality(self) -> Tuple[Dict[str, SemanticMetrics], List[Dict]]:
        """
        Evaluate semantic alignment between CRUD roles and database schemas.
        
        Uses embedding-based cosine similarity to measure how well role
        names/descriptions align with the database schema.
        
        Returns:
            Tuple of (Dict mapping db_id to SemanticMetrics, List of per-role detail records)
        """
        if not self.role_assignments:
            raise ValueError("Role assignments not loaded.")
        if not self.schema_summaries:
            # Return empty if no schema summaries provided
            return {}, []
        
        # Import embedding utilities
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plot"))
            from plot.embedding_utils import embed_texts, cosine_similarity as emb_cosine
        except ImportError:
            # Fallback: try direct import
            try:
                from embedding_utils import embed_texts, cosine_similarity as emb_cosine
            except ImportError:
                print("Warning: embedding_utils not available, skipping semantic evaluation")
                return {}, []
        
        # Prepare texts for embedding
        role_records = []
        schema_texts = set()
        role_texts = set()
        
        for db_id, roles_list in self.role_assignments.items():
            schema_text = self.schema_summaries.get(db_id, "")
            if not schema_text:
                continue
            
            schema_texts.add(schema_text)
            
            for role in roles_list:
                role_text = self._format_role_text(role)
                role_texts.add(role_text)
                
                is_sm = role.get('role', '').strip().lower() == 'systemmanager'
                
                # Count accessible tables/permissions
                tables_with_access = set()
                if role.get('INSERT'):
                    tables_with_access.update(role['INSERT'])
                if role.get('DELETE'):
                    tables_with_access.update(role['DELETE'])
                if role.get('tables'):
                    tables_with_access.update(role['tables'].keys())
                
                role_records.append({
                    'db_id': db_id,
                    'role': role.get('role', ''),
                    'schema_text': schema_text,
                    'role_text': role_text,
                    'is_system_manager': is_sm,
                    'num_tables': len(tables_with_access)
                })
        
        if not role_records:
            return {}, []
        
        # Compute embeddings using configured model
        EMBED_MODEL = self.embed_model
        EMBED_BACKEND = self.embed_backend
        
        all_texts = list(schema_texts) + list(role_texts)
        try:
            embeddings = embed_texts(
                all_texts,
                model_name=EMBED_MODEL,
                backend=EMBED_BACKEND
            )
            text_to_emb = {text: emb for text, emb in zip(all_texts, embeddings)}
        except Exception as e:
            print(f"Warning: Failed to compute embeddings: {e}")
            return {}, []
        
        # Compute similarity scores and collect per-role details
        metrics = {}
        role_details = []  # Per-role data for visualization
        
        for db_id in sorted(set(r['db_id'] for r in role_records)):
            db_roles = [r for r in role_records if r['db_id'] == db_id]
            
            similarities = []
            
            for role_rec in db_roles:
                schema_emb = text_to_emb.get(role_rec['schema_text'])
                role_emb = text_to_emb.get(role_rec['role_text'])
                
                # Compute similarity (normalized to [0, 1])
                raw_sim = self.cosine_similarity(role_emb, schema_emb)
                similarity = max(0.0, min(1.0, (raw_sim + 1.0) / 2.0))
                similarities.append(similarity)
                
                # Store per-role detail for visualization
                role_details.append({
                    'db_id': db_id,
                    'role': role_rec['role'],
                    'similarity': similarity,
                    'is_system_manager': role_rec['is_system_manager'],
                })
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            min_similarity = min(similarities) if similarities else 0.0
            max_similarity = max(similarities) if similarities else 0.0
            
            issues = []
            if avg_similarity < self.SIMILARITY_MIN:
                issues.append(f"Low semantic alignment ({avg_similarity:.2f} < {self.SIMILARITY_MIN})")
            
            metrics[db_id] = SemanticMetrics(
                db_id=db_id,
                avg_similarity=avg_similarity,
                min_similarity=min_similarity,
                max_similarity=max_similarity,
                num_roles=len(db_roles),
                issues=issues
            )
        
        return metrics, role_details
    
    def evaluate_all(self) -> CrudQualityReport:
        """
        Run all quality evaluations and generate comprehensive report.
        
        Returns:
            CrudQualityReport with all metrics and issue summary
        """
        report = CrudQualityReport()
        
        # Run evaluations
        if self.dataset:
            report.deny_rate_metrics = self.evaluate_deny_rates()
            report.total_entries = len(self.dataset)
            report.operation_distribution = self.get_operation_distribution()
        
        if self.role_assignments:
            report.coverage_metrics = self.evaluate_coverage()
            report.overlap_metrics = self.evaluate_policy_overlap()
            
            # Semantic evaluation (optional, requires schema_summaries)
            if self.schema_summaries:
                try:
                    semantic_metrics, role_details = self.evaluate_semantic_quality()
                    report.semantic_metrics = semantic_metrics
                    report.semantic_role_details = role_details
                except Exception as e:
                    print(f"Warning: Semantic evaluation failed: {e}")
            
            # Count totals
            report.total_databases = len(self.role_assignments)
            report.total_roles = sum(len(roles) for roles in self.role_assignments.values())
        
        # Aggregate all issues
        all_db_ids = set()
        if report.deny_rate_metrics:
            all_db_ids.update(report.deny_rate_metrics.keys())
        if report.coverage_metrics:
            all_db_ids.update(report.coverage_metrics.keys())
        if report.overlap_metrics:
            all_db_ids.update(report.overlap_metrics.keys())
        if report.semantic_metrics:
            all_db_ids.update(report.semantic_metrics.keys())
        
        for db_id in sorted(all_db_ids):
            issues = []
            
            if db_id in report.deny_rate_metrics and report.deny_rate_metrics[db_id].issues:
                issues.extend(report.deny_rate_metrics[db_id].issues)
            if db_id in report.coverage_metrics and report.coverage_metrics[db_id].issues:
                issues.extend(report.coverage_metrics[db_id].issues)
            if db_id in report.overlap_metrics and report.overlap_metrics[db_id].issues:
                issues.extend(report.overlap_metrics[db_id].issues)
            if db_id in report.semantic_metrics and report.semantic_metrics[db_id].issues:
                issues.extend(report.semantic_metrics[db_id].issues)
            
            if issues:
                report.all_issues[db_id] = issues
                report.databases_with_issues.append(db_id)
        
        return report


def print_crud_quality_report(report: CrudQualityReport, verbose: bool = True):
    """
    Print formatted quality report for CRUD-level RBAC.
    
    Args:
        report: CrudQualityReport from evaluator
        verbose: Whether to print detailed per-database info
    """
    print("=" * 90)
    print("CRUD-LEVEL ROLE GENERATION QUALITY REPORT")
    print("=" * 90)
    
    # Summary statistics
    print(f"\nOverall Statistics:")
    print(f"   Total databases: {report.total_databases}")
    print(f"   Total roles: {report.total_roles}")
    print(f"   Total entries: {report.total_entries}")
    print(f"   Databases with issues: {len(report.databases_with_issues)}")
    print(f"   Pass rate: {report.pass_rate:.1%}")
    
    # Operation distribution
    if report.operation_distribution:
        print("\nOperation Distribution:")
        total_ops = sum(report.operation_distribution.values())
        for op, count in sorted(report.operation_distribution.items()):
            pct = count / total_ops * 100 if total_ops > 0 else 0
            print(f"   {op}: {count} ({pct:.1f}%)")
    
    # Deny rate summary
    if report.deny_rate_metrics and verbose:
        print("\n" + "-" * 90)
        print("Deny Rate Analysis (by Operation Type)")
        print("-" * 90)
        print(f"{'Database':<20} {'Overall':>8} {'DDL':>8} {'INSERT':>8} {'DELETE':>8} {'UPDATE':>8} {'Status':>8}")
        print("-" * 90)
        
        for db_id, m in sorted(report.deny_rate_metrics.items()):
            status = "[WARN]" if m.issues else "[OK]"
            print(f"{db_id:<20} {m.deny_rate:>7.1f}% {m.ddl_deny_rate:>7.1f}% "
                  f"{m.insert_deny_rate:>7.1f}% {m.delete_deny_rate:>7.1f}% "
                  f"{m.update_deny_rate:>7.1f}% {status:>8}")
    
    # Coverage summary
    if report.coverage_metrics and verbose:
        print("\n" + "-" * 90)
        print("Permission Coverage Analysis")
        print("-" * 90)
        print(f"{'Database':<20} {'Tables':>8} {'INSERT':>10} {'DELETE':>10} {'UPDATE':>10} {'Diversity':>10}")
        print("-" * 90)
        
        for db_id, m in sorted(report.coverage_metrics.items()):
            status = "[WARN]" if m.issues else ""
            print(f"{db_id:<20} {m.total_tables:>8} {m.insert_coverage:>9.1%} "
                  f"{m.delete_coverage:>9.1%} {m.update_coverage:>9.1%} "
                  f"{m.coverage_std:>9.2f} {status}")
    
    # Policy overlap summary
    if report.overlap_metrics and verbose:
        print("\n" + "-" * 90)
        print("Policy Overlap Analysis")
        print("-" * 90)
        print(f"{'Database':<20} {'Avg Overlap':>12} {'Max Overlap':>12} {'Status':>10}")
        print("-" * 90)
        
        for db_id, m in sorted(report.overlap_metrics.items()):
            status = "[WARN]" if m.issues else "[OK]"
            print(f"{db_id:<20} {m.avg_overlap:>11.1%} {m.max_overlap:>11.1%} {status:>10}")
    
    # Issues summary
    print("\n" + "=" * 90)
    print("Issues Summary")
    print("=" * 90)
    
    if report.all_issues:
        for db_id, issues in report.all_issues.items():
            print(f"\n[WARN] {db_id}:")
            for issue in issues:
                print(f"   - {issue}")
    else:
        print("\n[OK] No quality issues detected!")
    
    print("\n" + "=" * 90)

