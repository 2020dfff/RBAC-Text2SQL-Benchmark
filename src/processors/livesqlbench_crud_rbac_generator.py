"""
CRUD-Level RBAC Dataset Generator for LiveSQLBench-Full PostgreSQL.

This module generates CRUD-level (SELECT, INSERT, UPDATE, DELETE, DDL) 
RBAC datasets for LiveSQLBench-Full, which includes both Query and 
Management tasks requiring different permission levels.

Key features:
1. Parse PostgreSQL schema from SQL dump files
2. Use sqlglot-based permission extraction for ground truth
3. Generate CRUD-level RBAC datasets with full permission checking
4. Support INSERT, UPDATE, DELETE, DDL operations (not just SELECT)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .sql_permission_extractor import SQLPermissionExtractor, PermissionResult


# ===========================================================================
# Constants
# ===========================================================================

DENIAL_MESSAGE = "Sorry, I cannot answer."

# LiveSQLBench operation types
OPERATION_TYPES = ["SELECT", "INSERT", "UPDATE", "DELETE", "DDL"]

# Instruction template for CRUD operations
CRUD_INSTRUCTION_TEMPLATE = """##Instruction:
Database: {db_id}
Total Tables: {total_tables}
Total Columns: {total_columns}

Table Schemas:

{schema_text}

##Role Access Policy (CRUD-Level):
Role: {role}
{policy_text}

Based on the above schema and your role's CRUD-level access policy, generate a SQL query to answer the question. 
If your role does not have the required permissions (SELECT, INSERT, UPDATE, DELETE, or DDL), respond with: "Sorry, I cannot answer."
"""


# ===========================================================================
# Schema Parsing for PostgreSQL
# ===========================================================================

def parse_postgresql_schema(schema_content: str) -> Dict[str, List[str]]:
    """
    Parse PostgreSQL schema to extract table → columns mapping.
    
    Handles CREATE TABLE statements with column definitions.
    """
    schema = {}
    
    # Pattern for CREATE TABLE with columns
    create_table_pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\']?(\w+)["\']?\s*\((.*?)\);',
        re.IGNORECASE | re.DOTALL
    )
    
    # Simple column pattern (handles most cases)
    column_pattern = re.compile(
        r'^\s*["\']?(\w+)["\']?\s+(?:INTEGER|BIGINT|SMALLINT|SERIAL|TEXT|VARCHAR|CHAR|BOOLEAN|DATE|TIMESTAMP|TIME|NUMERIC|DECIMAL|REAL|DOUBLE|FLOAT|JSON|JSONB|UUID|BYTEA|ARRAY|INT)',
        re.IGNORECASE | re.MULTILINE
    )
    
    for match in create_table_pattern.finditer(schema_content):
        table_name = match.group(1).lower()
        columns_block = match.group(2)
        
        columns = []
        for col_match in column_pattern.finditer(columns_block):
            col_name = col_match.group(1).lower()
            if col_name not in ('primary', 'foreign', 'unique', 'check', 'constraint', 'references'):
                columns.append(col_name)
        
        # Also try simpler extraction for column definitions
        if not columns:
            for line in columns_block.split(','):
                line = line.strip()
                if line and not line.upper().startswith(('PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'CONSTRAINT')):
                    parts = line.split()
                    if parts:
                        col_name = parts[0].strip('"\'').lower()
                        if col_name and col_name.isidentifier():
                            columns.append(col_name)
        
        if columns:
            schema[table_name] = columns
    
    return schema


def load_schema_from_file(schema_path: Path) -> Dict[str, List[str]]:
    """Load and parse schema from a SQL file."""
    if not schema_path.exists():
        return {}
    
    content = schema_path.read_text(encoding='utf-8')
    return parse_postgresql_schema(content)


def format_schema_text(schema: Dict[str, List[str]]) -> str:
    """Format schema dict to readable text."""
    parts = []
    for table, columns in sorted(schema.items()):
        parts.append(f"Table: {table}")
        parts.append(f"  Columns: {', '.join(columns)}")
    return '\n'.join(parts)


def format_crud_policy_text(role_def: Dict[str, Any]) -> str:
    """
    Format CRUD role definition to readable text.
    
    Expected role_def format:
    {
        "role": "HRManager",
        "description": "...",
        "DDL": false,
        "INSERT": ["employees"],
        "DELETE": [],
        "tables": {
            "employees": {"SELECT": ["*"], "UPDATE": ["name", "status"]}
        }
    }
    """
    parts = []
    
    # Global permissions
    parts.append(f"DDL Permission: {'YES' if role_def.get('DDL', False) else 'NO'}")
    
    insert_tables = role_def.get('INSERT', [])
    parts.append(f"INSERT Allowed On: {', '.join(insert_tables) if insert_tables else 'NONE'}")
    
    delete_tables = role_def.get('DELETE', [])
    parts.append(f"DELETE Allowed On: {', '.join(delete_tables) if delete_tables else 'NONE'}")
    
    # Per-table permissions
    parts.append("\nTable Permissions:")
    tables = role_def.get('tables', {})
    for table, perms in sorted(tables.items()):
        select_cols = perms.get('SELECT', [])
        update_cols = perms.get('UPDATE', [])
        
        select_str = 'ALL COLUMNS' if select_cols == ['*'] else ', '.join(select_cols) if select_cols else 'NONE'
        update_str = 'ALL COLUMNS' if update_cols == ['*'] else ', '.join(update_cols) if update_cols else 'NONE'
        
        parts.append(f"  {table}:")
        parts.append(f"    SELECT: {select_str}")
        parts.append(f"    UPDATE: {update_str}")
    
    return '\n'.join(parts)


# ===========================================================================
# Permission Checking
# ===========================================================================

def check_crud_permission(
    ground_truth: Dict[str, Any],
    role_def: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check if a SQL operation is allowed under the role's CRUD policy.
    
    Args:
        ground_truth: Required permissions from livesqlbench_permission_ground_truth.json
        role_def: Role definition with CRUD permissions
    
    Returns:
        (allowed, denial_reason, missing_permissions)
    """
    required = ground_truth['required_permissions']
    missing = {'read': {}, 'write': {}, 'insert': [], 'delete': [], 'ddl': []}
    
    # 1. Check DDL permission
    if required.get('ddl'):
        if not role_def.get('DDL', False):
            missing['ddl'] = required['ddl']
            return False, 'ddl_not_allowed', missing
    
    # 2. Check INSERT permission  
    for table in required.get('insert', []):
        if table not in role_def.get('INSERT', []):
            missing['insert'].append(table)
    if missing['insert']:
        return False, 'insert_not_allowed', missing
    
    # 3. Check DELETE permission
    for table in required.get('delete', []):
        if table not in role_def.get('DELETE', []):
            missing['delete'].append(table)
    if missing['delete']:
        return False, 'delete_not_allowed', missing
    
    # 4. Check WRITE/UPDATE permission
    tables_config = role_def.get('tables', {})
    for table, cols in required.get('write', {}).items():
        table_perms = tables_config.get(table, {})
        allowed_update = set(table_perms.get('UPDATE', []))
        if '*' not in allowed_update:
            missing_cols = [c for c in cols if c not in allowed_update]
            if missing_cols:
                missing['write'][table] = missing_cols
    if missing['write']:
        return False, 'write_not_allowed', missing
    
    # 5. Check READ/SELECT permission
    for table, cols in required.get('read', {}).items():
        table_perms = tables_config.get(table, {})
        allowed_select = set(table_perms.get('SELECT', []))
        if '*' not in allowed_select:
            missing_cols = [c for c in cols if c not in allowed_select]
            if missing_cols:
                missing['read'][table] = missing_cols
    if missing['read']:
        return False, 'read_not_allowed', missing
    
    return True, None, {}


# ===========================================================================
# Main Generator Class
# ===========================================================================

@dataclass
class LiveSQLBenchCRUDEntry:
    """A single entry in the CRUD-level RBAC dataset."""
    instance_id: str
    db_id: str
    question: str
    role: str
    policy: Dict[str, Any]
    output: str
    gold_sql: str
    operation: str
    allowed: bool
    denial_reason: Optional[str] = None
    instruction: str = ""
    input: str = ""


class LiveSQLBenchCRUDRBACGenerator:
    """
    Generator for CRUD-level RBAC dataset for LiveSQLBench-Full.
    
    Usage:
        generator = LiveSQLBenchCRUDRBACGenerator(
            data_dir='data/livesqlbench-full-postgresql',
            oracle=oracle
        )
        
        # Generate roles for databases
        roles = generator.generate_roles_for_databases(db_ids)
        
        # Generate RBAC dataset
        dataset = generator.generate_dataset(roles)
    """
    
    def __init__(
        self,
        data_dir: str,
        oracle=None,
        max_workers: int = 4
    ):
        """
        Initialize the generator.
        
        Args:
            data_dir: Path to livesqlbench-full-postgresql data directory
            oracle: LLM Oracle instance for role generation
            max_workers: Number of parallel workers for batch processing
        """
        self.data_dir = Path(data_dir)
        self.oracle = oracle
        self.max_workers = max_workers
        self.extractor = SQLPermissionExtractor(dialect="postgres")
        
        # Load ground truth
        self.ground_truth_path = self.data_dir / 'livesqlbench_permission_ground_truth.json'
        self.ground_truths = {}
        self._load_ground_truth()
        
        # Load original data
        self.data_path = self.data_dir / 'livesqlbench_base_full_v1_gt_kg_testcases_0904.jsonl'
        self.raw_data = []
        self._load_raw_data()
    
    def _load_ground_truth(self):
        """Load pre-computed permission ground truth."""
        if self.ground_truth_path.exists():
            with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for gt in data.get('ground_truths', []):
                self.ground_truths[gt['instance_id']] = gt
            print(f"Loaded {len(self.ground_truths)} ground truth entries")
        else:
            print(f"Warning: Ground truth file not found at {self.ground_truth_path}")
    
    def _load_raw_data(self):
        """Load original LiveSQLBench data and merge with ground truth."""
        if self.data_path.exists():
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        instance_id = entry.get('instance_id')
                        
                        # Get db_id and other info from ground truth
                        if instance_id and instance_id in self.ground_truths:
                            gt = self.ground_truths[instance_id]
                            entry['db_id'] = gt.get('db_id', '')
                            entry['operation'] = gt.get('operation', 'SELECT')
                        
                        # Extract gold_sql from sol_sql (it's a list)
                        sol_sql = entry.get('sol_sql', [])
                        if isinstance(sol_sql, list) and sol_sql:
                            entry['gold_sql'] = sol_sql[0]  # Take the first SQL
                        elif isinstance(sol_sql, str):
                            entry['gold_sql'] = sol_sql
                        else:
                            entry['gold_sql'] = ''
                        
                        # Use instance_id based question placeholder (external_knowledge is just index)
                        entry['question'] = f"Query for {instance_id}"
                        
                        self.raw_data.append(entry)
            print(f"Loaded {len(self.raw_data)} raw data entries")
        else:
            print(f"Warning: Data file not found at {self.data_path}")
    
    def get_database_ids(self) -> List[str]:
        """Get unique database IDs from the ground truth."""
        return sorted(set(gt.get('db_id', '') for gt in self.ground_truths.values() if gt.get('db_id')))
    
    def get_database_schema(self, db_id: str) -> Dict[str, List[str]]:
        """
        Get schema for a database.
        
        Looks for schema in:
        1. bird-interact-full-dumps/{db_id}/schema.sql
        2. Extract from data entries
        """
        # Try schema file
        schema_dir = self.data_dir / 'bird-interact-full-dumps' / db_id
        schema_file = schema_dir / 'schema.sql'
        
        if schema_file.exists():
            schema = load_schema_from_file(schema_file)
            if schema:
                return schema
        
        # Fallback: extract from ground truth (collect all columns for each table)
        schema = {}
        for gt in self.ground_truths.values():
            if gt['db_id'] == db_id:
                # Collect columns from read permissions
                for table, cols in gt['required_permissions'].get('read', {}).items():
                    if table not in schema:
                        schema[table] = set()
                    schema[table].update(cols)
                
                # Collect columns from write permissions  
                for table, cols in gt['required_permissions'].get('write', {}).items():
                    if table not in schema:
                        schema[table] = set()
                    schema[table].update(cols)
                
                # Add tables from insert/delete (no columns available)
                for table in gt['required_permissions'].get('insert', []):
                    if table not in schema:
                        schema[table] = set()
                for table in gt['required_permissions'].get('delete', []):
                    if table not in schema:
                        schema[table] = set()
        
        # Convert sets to sorted lists, use ['*'] if no columns found
        for table in schema:
            if schema[table]:
                schema[table] = sorted(list(schema[table]))
            else:
                schema[table] = ['*']
        
        return schema
    
    def generate_roles_for_database(
        self,
        db_id: str,
        system_prompt: str,
        user_prompt_template: str,
        max_tokens: int = 4000
    ) -> List[Dict[str, Any]]:
        """
        Generate CRUD-level roles for a database using LLM.
        
        Returns list of role definitions.
        """
        if not self.oracle:
            raise ValueError("Oracle not initialized - cannot generate roles")
        
        schema = self.get_database_schema(db_id)
        if not schema:
            print(f"Warning: No schema found for {db_id}")
            return []
        
        schema_text = format_schema_text(schema)
        user_prompt = user_prompt_template.format(schema_content=schema_text)
        
        response, usage = self.oracle.query(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )
        
        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                roles = json.loads(json_match.group())
                return roles
            else:
                print(f"Warning: No JSON found in response for {db_id}")
                return []
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse roles for {db_id}: {e}")
            return []
    
    def generate_roles_batch(
        self,
        db_ids: List[str],
        system_prompt: str,
        user_prompt_template: str,
        max_tokens: int = 4000
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate roles for multiple databases in parallel.
        
        Returns {db_id: [role_definitions]}
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.generate_roles_for_database,
                    db_id, system_prompt, user_prompt_template, max_tokens
                ): db_id
                for db_id in db_ids
            }
            
            for future in as_completed(futures):
                db_id = futures[future]
                try:
                    roles = future.result()
                    results[db_id] = roles
                    print(f"Generated {len(roles)} roles for {db_id}")
                except Exception as e:
                    print(f"Error generating roles for {db_id}: {e}")
                    results[db_id] = []
        
        return results
    
    def generate_dataset_entry(
        self,
        instance_id: str,
        role_def: Dict[str, Any],
        schema: Dict[str, List[str]]
    ) -> Optional[LiveSQLBenchCRUDEntry]:
        """
        Generate a single RBAC dataset entry.
        
        Checks if the role has permission for the SQL operation.
        """
        # Get raw data
        raw_entry = next(
            (item for item in self.raw_data if item.get('instance_id') == instance_id),
            None
        )
        if not raw_entry:
            return None
        
        # Get ground truth
        gt = self.ground_truths.get(instance_id)
        if not gt:
            return None
        
        db_id = raw_entry.get('db_id', '')
        question = raw_entry.get('question', '')
        gold_sql = raw_entry.get('gold_sql', '')
        operation = gt.get('operation', 'SELECT')
        
        # Check permission
        allowed, denial_reason, _ = check_crud_permission(gt, role_def)
        
        # Build instruction
        schema_text = format_schema_text(schema)
        policy_text = format_crud_policy_text(role_def)
        
        instruction = CRUD_INSTRUCTION_TEMPLATE.format(
            db_id=db_id,
            total_tables=len(schema),
            total_columns=sum(len(cols) for cols in schema.values()),
            schema_text=schema_text,
            role=role_def.get('role', 'Unknown'),
            policy_text=policy_text
        )
        
        # Determine output
        output = gold_sql if allowed else DENIAL_MESSAGE
        
        return LiveSQLBenchCRUDEntry(
            instance_id=instance_id,
            db_id=db_id,
            question=question,
            role=role_def.get('role', 'Unknown'),
            policy=role_def,
            output=output,
            gold_sql=gold_sql,
            operation=operation,
            allowed=allowed,
            denial_reason=denial_reason,
            instruction=instruction,
            input=question
        )
    
    def generate_dataset(
        self,
        role_assignments: Dict[str, List[Dict[str, Any]]],
        include_all_roles: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate the full RBAC dataset.
        
        Args:
            role_assignments: {db_id: [role_definitions]}
            include_all_roles: If True, each query is paired with ALL roles
        
        Returns:
            List of dataset entries as dicts
        """
        dataset = []
        
        for raw_entry in self.raw_data:
            instance_id = raw_entry.get('instance_id')
            db_id = raw_entry.get('db_id')
            
            if not instance_id or not db_id:
                continue
            
            # Skip if no ground truth
            if instance_id not in self.ground_truths:
                continue
            
            # Get roles for this database
            roles = role_assignments.get(db_id, [])
            if not roles:
                continue
            
            # Get schema
            schema = self.get_database_schema(db_id)
            
            # Generate entry for each role
            for role_def in roles:
                entry = self.generate_dataset_entry(instance_id, role_def, schema)
                if entry:
                    dataset.append(asdict(entry))
        
        return dataset
    
    def save_roles(self, roles: Dict[str, List[Dict]], output_path: str):
        """Save role assignments to JSON file."""
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_databases': len(roles),
                'total_roles': sum(len(r) for r in roles.values())
            },
            'assignments': roles
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Saved roles to {output_path}")
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """Save RBAC dataset to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Saved dataset with {len(dataset)} entries to {output_path}")
    
    def get_statistics(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Get statistics for a generated dataset."""
        stats = {
            'total_entries': len(dataset),
            'allowed': sum(1 for e in dataset if e.get('allowed')),
            'denied': sum(1 for e in dataset if not e.get('allowed')),
            'by_operation': {},
            'by_database': {},
            'by_role': {},
            'denial_reasons': {}
        }
        
        for entry in dataset:
            op = entry.get('operation', 'UNKNOWN')
            db = entry.get('db_id', 'UNKNOWN')
            role = entry.get('role', 'UNKNOWN')
            reason = entry.get('denial_reason', None)
            
            # By operation
            if op not in stats['by_operation']:
                stats['by_operation'][op] = {'allowed': 0, 'denied': 0}
            if entry.get('allowed'):
                stats['by_operation'][op]['allowed'] += 1
            else:
                stats['by_operation'][op]['denied'] += 1
            
            # By database
            if db not in stats['by_database']:
                stats['by_database'][db] = {'total': 0, 'allowed': 0}
            stats['by_database'][db]['total'] += 1
            if entry.get('allowed'):
                stats['by_database'][db]['allowed'] += 1
            
            # By role
            if role not in stats['by_role']:
                stats['by_role'][role] = {'allowed': 0, 'denied': 0}
            if entry.get('allowed'):
                stats['by_role'][role]['allowed'] += 1
            else:
                stats['by_role'][role]['denied'] += 1
            
            # Denial reasons
            if reason:
                stats['denial_reasons'][reason] = stats['denial_reasons'].get(reason, 0) + 1
        
        return stats


# ===========================================================================
# Deterministic Role Generation (Fallback without LLM)
# ===========================================================================

def generate_deterministic_crud_roles(schema: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Generate deterministic CRUD roles for testing without LLM.
    
    Creates a hierarchy with realistic access restrictions:
    1. ReadOnlyAnalyst - SELECT on ~50% of tables (limited columns)
    2. DataEditor - SELECT all + UPDATE on ~30% of tables
    3. DataManager - SELECT + UPDATE + INSERT + DELETE (no DDL)
    4. SystemManager - Full access including DDL
    """
    all_tables = list(schema.keys())
    
    if not all_tables:
        return []
    
    roles = []
    
    # Split tables for different access levels
    mid_point = max(1, len(all_tables) // 2)
    third_point = max(1, len(all_tables) // 3)
    
    readonly_table_names = all_tables[:mid_point]  # First half
    editor_update_tables = all_tables[:third_point]  # First third can be updated
    
    # 1. ReadOnlyAnalyst - SELECT on limited tables, limited columns
    readonly_tables = {}
    for table in readonly_table_names:
        cols = schema.get(table, ['*'])
        if cols and cols != ['*']:
            # Only allow first 5 columns
            readonly_tables[table] = {'SELECT': cols[:5], 'UPDATE': []}
        else:
            readonly_tables[table] = {'SELECT': ['*'], 'UPDATE': []}
    
    roles.append({
        'role': 'ReadOnlyAnalyst',
        'description': 'Limited read access for basic analytics',
        'DDL': False,
        'INSERT': [],
        'DELETE': [],
        'tables': readonly_tables
    })
    
    # 2. DataEditor - SELECT all + UPDATE limited tables
    editor_tables = {}
    for table, cols in schema.items():
        if table in editor_update_tables:
            # Can UPDATE non-id columns on limited tables
            if cols and cols != ['*']:
                update_cols = [c for c in cols if not c.endswith('id') and c != 'id'][:3]
            else:
                update_cols = []
            editor_tables[table] = {
                'SELECT': cols if cols != ['*'] else ['*'],
                'UPDATE': update_cols
            }
        else:
            # Read-only on other tables
            editor_tables[table] = {
                'SELECT': cols if cols != ['*'] else ['*'],
                'UPDATE': []
            }
    
    roles.append({
        'role': 'DataEditor',
        'description': 'Can read all and update specific columns on selected tables',
        'DDL': False,
        'INSERT': [],
        'DELETE': [],
        'tables': editor_tables
    })
    
    # 3. DataManager - Full data manipulation (no DDL)
    manager_tables = {}
    for table, cols in schema.items():
        manager_tables[table] = {'SELECT': ['*'], 'UPDATE': ['*']}
    
    roles.append({
        'role': 'DataManager',
        'description': 'Full data manipulation including insert and delete',
        'DDL': False,
        'INSERT': all_tables,
        'DELETE': all_tables,
        'tables': manager_tables
    })
    
    # 4. SystemManager - Full access
    admin_tables = {}
    for table in schema.keys():
        admin_tables[table] = {'SELECT': ['*'], 'UPDATE': ['*']}
    
    roles.append({
        'role': 'SystemManager',
        'description': 'Full administrative access including DDL operations',
        'DDL': True,
        'INSERT': all_tables,
        'DELETE': all_tables,
        'tables': admin_tables
    })
    
    return roles
