"""
Role Generation Quality Metrics

This module provides reusable quality assessment utilities for evaluating
role generation quality in RBAC datasets. It includes:

1. Deny Rate Analysis - Evaluate permission distribution balance
2. Semantic Similarity Analysis - Embedding-based role-schema alignment
3. Policy Overlap Analysis - Jaccard similarity between role policies

Usage:
    from src.role_evaluator.quality_metrics import RoleQualityEvaluator
    
    evaluator = RoleQualityEvaluator(
        role_assignments_path="outputs/column_level_role_assignments_bird.json",
        dataset_path="outputs/column_level_rbac_dataset_bird.json",
        schema_dir="data/Bird/dev_20251106/dev_databases"
    )
    
    # Run all evaluations
    report = evaluator.evaluate_all()
    
    # Or run individual evaluations
    deny_report = evaluator.evaluate_deny_rates()
    semantic_report = evaluator.evaluate_semantic_quality()
    overlap_report = evaluator.evaluate_policy_overlap()
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
    """Metrics for semantic quality per database."""
    db_id: str
    avg_similarity: float
    avg_coverage: float
    coverage_std: float
    num_roles: int
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
    
    # Quality thresholds
    DENY_RATE_MIN = 5.0    # Minimum acceptable deny rate (%)
    DENY_RATE_MAX = 90.0   # Maximum acceptable deny rate (%)
    # Note: We don't flag high all-allow as an issue - complex DBs naturally have lower all-allow
    SIMILARITY_MIN = 0.6   # Minimum semantic similarity threshold
    OVERLAP_MAX = 0.7      # Maximum acceptable policy overlap (Jaccard similarity)
    COVERAGE_STD_MIN = 0.1 # Minimum coverage std dev for diversity
    
    def __init__(
        self,
        role_assignments_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        schema_dir: Optional[str] = None,
        role_assignments: Optional[Dict] = None,
        dataset: Optional[List[Dict]] = None,
        schema_summaries: Optional[Dict] = None,
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
        """
        self.role_assignments = role_assignments
        self.dataset = dataset
        self.schema_summaries = schema_summaries
        
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
        """Load schema summaries from *_schema_detailed.txt files or SQLite databases."""
        schema_path = Path(schema_dir)
        self.schema_summaries = {}
        
        for db_dir in sorted(schema_path.iterdir()):
            if not db_dir.is_dir() or db_dir.name.startswith('.'):
                continue
            
            db_id = db_dir.name
            schema_files = list(db_dir.glob('*_schema_detailed.txt'))
            
            table_columns = {}  # {table_name: column_count}
            table_count = 0
            column_count = 0
            schema_text = ""
            
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
            
            # Fallback: query SQLite database directly if schema file doesn't have per-table info
            if not table_columns:
                db_file = db_dir / f"{db_id}.sqlite"
                if db_file.exists():
                    try:
                        import sqlite3
                        conn = sqlite3.connect(str(db_file))
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                        tables = [r[0] for r in cursor.fetchall()]
                        for table in tables:
                            cursor.execute(f"PRAGMA table_info(`{table}`)")
                            cols = cursor.fetchall()
                            table_columns[table] = len(cols)
                        conn.close()
                        table_count = len(tables)
                        column_count = sum(table_columns.values())
                    except Exception:
                        pass
            
            self.schema_summaries[db_id] = {
                "db_id": db_id,
                "table_count": table_count,
                "column_count": column_count,
                "table_columns": table_columns,  # NEW: per-table column counts
                "summary_text": schema_text,
            }
    
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
        
        # Prepare role records
        role_records = self._prepare_role_records()
        
        # Get unique texts
        unique_schema_texts = list(set(
            spec["schema_text"] for spec in role_records if spec["schema_text"]
        ))
        unique_role_texts = list(set(spec["role_text"] for spec in role_records))
        
        # Compute embeddings
        if unique_schema_texts:
            schema_embeddings = embed_texts(unique_schema_texts)
            self._schema_emb_dict = {
                text: emb for text, emb in zip(unique_schema_texts, schema_embeddings)
            }
        
        if unique_role_texts:
            role_embeddings = embed_texts(unique_role_texts)
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
            
            issues = []
            if avg_similarity < self.SIMILARITY_MIN:
                issues.append(f"Low semantic alignment ({avg_similarity:.2f} < {self.SIMILARITY_MIN})")
            if coverage_std < self.COVERAGE_STD_MIN and len(non_sm_roles) > 1:
                issues.append("Low role diversity")
            if avg_coverage > 0.9 and len(non_sm_roles) > 0:
                issues.append("Most roles too permissive")
            
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
                
                # High overlap = roles are too similar = low diversity = redundant roles
                # This reduces the discriminative power of the RBAC dataset
                issues = []
                if max_overlap > self.OVERLAP_MAX:
                    issues.append(f"High max overlap ({max_overlap:.0%} > {self.OVERLAP_MAX:.0%}) between {max_pair['role1']} and {max_pair['role2']} - consider differentiating these roles")
                
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
