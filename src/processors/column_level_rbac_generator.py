"""
Column-Level RBAC Dataset Generator for BIRD Benchmark.

Generates role-aware SQL datasets using column-level permissions.
"""

import json
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .column_extractor import SQLColumnExtractor, extract_query_columns
from .column_permission_checker import ColumnPermissionChecker, check_column_permission
from .bird_describer import BirdDatabaseDescriber


# Denial message template - keep it simple to prevent inference attacks
DENIAL_MESSAGE = "Sorry, I cannot answer."

# Instruction template
INSTRUCTION_TEMPLATE = """##Instruction:
Database: {db_id}
Total Tables: {total_tables}
Total Columns: {total_columns}

Table Schemas:

{schema_text}

##Evidence:
{evidence}

##Role Access Policy (Column-Level):
Role: {role}
Accessible Columns: {policy_text}

Based on the above schema and your role's column-level access policy, generate a SQL query to answer the question. If your role does not have access to the required columns, respond with: "Sorry, I cannot answer."
"""


def format_policy_text(policy: Dict[str, List[str]]) -> str:
    """Format policy dict to readable text."""
    parts = []
    for table, columns in policy.items():
        if columns == ["*"] or columns == "*":
            parts.append(f"{table}: ALL COLUMNS")
        else:
            parts.append(f"{table}: {', '.join(columns)}")
    return "; ".join(parts)


@dataclass 
class RBACDatasetEntry:
    """A single entry in the RBAC dataset (matches table-level format)."""
    db_id: str
    instruction: str  # Full instruction with schema and policy
    role: str
    policy: Dict[str, List[str]]  # Column-level policy: {table: [columns]}
    input: str  # User question
    output: str  # Either SQL or denial message
    difficulty: str
    metadata: Dict[str, Any]  # Gold SQL, permission status, etc.
    
    def to_dict(self) -> dict:
        return asdict(self)


class ColumnLevelRBACGenerator:
    """Generate column-level RBAC datasets."""
    
    def __init__(
        self,
        role_assignments_path: str,
        bird_data_path: str,
        bird_db_root: str,
        project_root: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize generator.
        
        Args:
            role_assignments_path: Path to column_level_role_assignments_bird.json
            bird_data_path: Path to BIRD dev.json
            bird_db_root: Root directory containing BIRD databases
            project_root: Project root for BirdDatabaseDescriber
            seed: Random seed for reproducibility
        """
        self.role_assignments_path = Path(role_assignments_path)
        self.bird_data_path = Path(bird_data_path)
        self.bird_db_root = Path(bird_db_root)
        self.project_root = Path(project_root) if project_root else self.bird_db_root.parent.parent.parent
        self.seed = seed
        
        random.seed(seed)
        
        # Load data
        self.role_assignments = self._load_role_assignments()
        self.bird_data = self._load_bird_data()
        
        # Index queries by db_id
        self.queries_by_db = self._index_queries_by_db()
        
        # Initialize components
        self.checker = ColumnPermissionChecker()
        self.describer = BirdDatabaseDescriber(self.project_root)
        
        # Cache for database schemas
        self._schema_cache: Dict[str, str] = {}
        self._db_info_cache: Dict[str, Any] = {}
    
    def _load_role_assignments(self) -> Dict:
        """Load role assignments from JSON."""
        with open(self.role_assignments_path, 'r') as f:
            return json.load(f)
    
    def _load_bird_data(self) -> List[Dict]:
        """Load BIRD dataset."""
        with open(self.bird_data_path, 'r') as f:
            return json.load(f)
    
    def _index_queries_by_db(self) -> Dict[str, List[Dict]]:
        """Index queries by database ID."""
        index = {}
        for item in self.bird_data:
            db_id = item.get('db_id', '')
            if db_id not in index:
                index[db_id] = []
            index[db_id].append(item)
        return index
    
    def get_db_path(self, db_id: str) -> str:
        """Get full path to database file."""
        return str(self.bird_db_root / db_id / f"{db_id}.sqlite")
    
    def generate_dataset(
        self,
        roles_per_db: Optional[int] = None,
        queries_per_role: Optional[int] = None,
        balance_ratio: float = 0.5
    ) -> List[RBACDatasetEntry]:
        """
        Generate RBAC dataset.
        
        Args:
            roles_per_db: Max roles to use per database (None = use all roles)
            queries_per_role: Queries per role (None = all queries)
            balance_ratio: Target ratio of allowed vs denied (0.5 = balanced)
        
        Returns:
            List of dataset entries
        """
        entries = []
        
        assignments = self.role_assignments.get('assignments', {})
        
        for db_id, roles in assignments.items():
            if db_id not in self.queries_by_db:
                print(f"Warning: No queries found for database {db_id}")
                continue
            
            queries = self.queries_by_db[db_id]
            db_path = self.get_db_path(db_id)
            
            # Select roles to use (all roles if roles_per_db is None)
            if roles_per_db is not None and len(roles) > roles_per_db:
                selected_roles = roles[:roles_per_db]
            else:
                selected_roles = roles
            
            for role_info in selected_roles:
                role_entries = self._generate_entries_for_role(
                    db_id=db_id,
                    role_info=role_info,
                    queries=queries,
                    db_path=db_path,
                    max_queries=queries_per_role
                )
                entries.extend(role_entries)
        
        # Optionally balance the dataset
        if balance_ratio is not None:
            entries = self._balance_dataset(entries, balance_ratio)
        
        return entries
    
    def _get_schema_text(self, db_id: str) -> Tuple[str, int, int]:
        """Get schema text for a database, with caching."""
        if db_id in self._schema_cache:
            db_info = self._db_info_cache[db_id]
            return self._schema_cache[db_id], db_info.total_tables, db_info.total_columns
        
        # Get database path
        db_path = self.bird_db_root / db_id
        
        # Read database description
        db_info = self.describer.read_database_descriptions(db_path)
        
        if db_info:
            schema_text = self.describer.generate_schema_description(db_info, format_type="detailed")
            self._db_info_cache[db_id] = db_info
            self._schema_cache[db_id] = schema_text
            return schema_text, db_info.total_tables, db_info.total_columns
        else:
            # Fallback: get schema from SQLite directly
            schema_text, total_tables, total_columns = self._get_schema_from_sqlite(db_id)
            # Create a simple db_info for caching
            from dataclasses import dataclass
            @dataclass
            class SimpleDBInfo:
                total_tables: int
                total_columns: int
            self._db_info_cache[db_id] = SimpleDBInfo(total_tables, total_columns)
            self._schema_cache[db_id] = schema_text
            return schema_text, total_tables, total_columns
    
    def _get_schema_from_sqlite(self, db_id: str) -> Tuple[str, int, int]:
        """Get schema directly from SQLite database."""
        db_file = self.bird_db_root / db_id / f"{db_id}.sqlite"
        
        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            
            lines = [f"Database: {db_id}", f"Total Tables: {len(tables)}", ""]
            total_columns = 0
            
            for table in tables:
                cursor.execute(f"PRAGMA table_info(`{table}`)")
                columns = cursor.fetchall()
                total_columns += len(columns)
                
                lines.append(f"Table: {table} ({len(columns)} columns)")
                lines.append("-" * 40)
                for col in columns:
                    col_name, col_type = col[1], col[2]
                    lines.append(f"• {col_name} ({col_type})")
                lines.append("")
            
            conn.close()
            return "\n".join(lines), len(tables), total_columns
        except Exception as e:
            return f"Database: {db_id}\nError reading schema: {e}", 0, 0

    def _generate_entries_for_role(
        self,
        db_id: str,
        role_info: Dict,
        queries: List[Dict],
        db_path: str,
        max_queries: Optional[int] = None
    ) -> List[RBACDatasetEntry]:
        """Generate entries for a single role."""
        entries = []
        
        role = role_info.get('role', 'Unknown')
        policy = role_info.get('policy', {})
        
        # Get schema text
        schema_text, total_tables, total_columns = self._get_schema_text(db_id)
        policy_text = format_policy_text(policy)
        
        # Select queries
        if max_queries and max_queries < len(queries):
            selected_queries = random.sample(queries, max_queries)
        else:
            selected_queries = queries
        
        for query_item in selected_queries:
            question = query_item.get('question', '')
            evidence = query_item.get('evidence', '')
            gold_sql = query_item.get('SQL', '')
            difficulty = query_item.get('difficulty', 'unknown')
            question_id = query_item.get('question_id', '')
            
            if not gold_sql:
                continue
            
            # Check permission
            result = check_column_permission(gold_sql, policy, db_path)
            
            # Determine output
            if result.allowed:
                output = gold_sql
                permission = "allowed"
            else:
                output = DENIAL_MESSAGE
                permission = "denied"
            
            # Build instruction with schema and policy
            instruction = INSTRUCTION_TEMPLATE.format(
                db_id=db_id,
                total_tables=total_tables,
                total_columns=total_columns,
                schema_text=schema_text,
                evidence=evidence if evidence else "N/A",
                role=role,
                policy_text=policy_text
            )
            
            # Build metadata
            metadata = {
                "gold_sql": gold_sql,
                "permission": permission,
                "query_columns": result.to_dict()["required_columns"],
                "missing_columns": result.to_dict()["missing_columns"],
                "reason": result.reason,
                "question_id": question_id,
                "evidence": evidence
            }
            
            entry = RBACDatasetEntry(
                db_id=db_id,
                instruction=instruction,
                role=role,
                policy=policy,
                input=question,
                output=output,
                difficulty=difficulty,
                metadata=metadata
            )
            
            entries.append(entry)
        
        return entries
    
    def _balance_dataset(
        self,
        entries: List[RBACDatasetEntry],
        target_ratio: float
    ) -> List[RBACDatasetEntry]:
        """Balance dataset to achieve target allowed/denied ratio."""
        allowed = [e for e in entries if e.metadata.get("permission") == "allowed"]
        denied = [e for e in entries if e.metadata.get("permission") == "denied"]
        
        current_ratio = len(allowed) / len(entries) if entries else 0
        print(f"Current ratio (allowed): {current_ratio:.2%}")
        print(f"  Allowed: {len(allowed)}, Denied: {len(denied)}")
        
        # For now, return all entries without balancing
        # Balancing logic can be added if needed
        return entries
    
    def generate_and_save(
        self,
        output_path: str,
        roles_per_db: int = 5,
        queries_per_role: Optional[int] = None,
        balance_ratio: float = 0.5
    ) -> Dict:
        """
        Generate dataset and save to file.
        
        Args:
            output_path: Path to save output JSON
            roles_per_db: Number of roles per database
            queries_per_role: Queries per role (None = all)
            balance_ratio: Target balance ratio
        
        Returns:
            Summary statistics
        """
        print(f"Generating column-level RBAC dataset...")
        print(f"  Roles per DB: {roles_per_db}")
        print(f"  Queries per role: {queries_per_role or 'all'}")
        
        entries = self.generate_dataset(
            roles_per_db=roles_per_db,
            queries_per_role=queries_per_role,
            balance_ratio=balance_ratio
        )
        
        # Calculate statistics
        allowed_count = sum(1 for e in entries if e.metadata.get("permission") == "allowed")
        denied_count = sum(1 for e in entries if e.metadata.get("permission") == "denied")
        
        # Group by database
        by_db = {}
        for e in entries:
            if e.db_id not in by_db:
                by_db[e.db_id] = {"allowed": 0, "denied": 0}
            perm = e.metadata.get("permission", "unknown")
            if perm in by_db[e.db_id]:
                by_db[e.db_id][perm] += 1
        
        # Build output - save as list (matching table-level format)
        output_data = [e.to_dict() for e in entries]
        
        # Save main data file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Save metadata separately
        meta_path = output_path.parent / f"{output_path.stem}_metadata.json"
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "source_role_assignments": str(self.role_assignments_path),
            "source_bird_data": str(self.bird_data_path),
            "seed": self.seed,
            "config": {
                "roles_per_db": roles_per_db,
                "queries_per_role": queries_per_role,
                "balance_ratio": balance_ratio
            },
            "statistics": {
                "total_entries": len(entries),
                "allowed": allowed_count,
                "denied": denied_count,
                "allowed_ratio": allowed_count / len(entries) if entries else 0,
                "by_database": by_db
            }
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset saved to {output_path}")
        print(f"Metadata saved to {meta_path}")
        print(f"  Total entries: {len(entries)}")
        print(f"  Allowed: {allowed_count} ({allowed_count/len(entries)*100:.1f}%)")
        print(f"  Denied: {denied_count} ({denied_count/len(entries)*100:.1f}%)")
        
        return metadata["statistics"]


def generate_column_level_rbac_dataset(
    role_assignments_path: str,
    bird_data_path: str,
    bird_db_root: str,
    output_path: str,
    project_root: Optional[str] = None,
    roles_per_db: int = 5,
    queries_per_role: Optional[int] = None,
    seed: int = 42
) -> Dict:
    """
    Convenience function to generate column-level RBAC dataset.
    
    Args:
        role_assignments_path: Path to role assignments JSON
        bird_data_path: Path to BIRD dev.json
        bird_db_root: Root of BIRD databases
        output_path: Output JSON path
        project_root: Project root for schema description
        roles_per_db: Roles to use per database
        queries_per_role: Queries per role
        seed: Random seed
    
    Returns:
        Statistics dictionary
    """
    generator = ColumnLevelRBACGenerator(
        role_assignments_path=role_assignments_path,
        bird_data_path=bird_data_path,
        bird_db_root=bird_db_root,
        project_root=project_root,
        seed=seed
    )
    
    return generator.generate_and_save(
        output_path=output_path,
        roles_per_db=roles_per_db,
        queries_per_role=queries_per_role
    )
