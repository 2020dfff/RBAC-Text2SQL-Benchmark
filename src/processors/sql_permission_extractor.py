"""
SQL Permission Extractor using sqlglot.

This module extracts permission requirements from SQL statements,
identifying which tables and columns require read, write, insert,
delete, or DDL permissions.

Supported SQL dialects: PostgreSQL, SQLite, MySQL, etc.
"""

import sqlglot
from sqlglot import exp
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum


class OperationType(Enum):
    """SQL operation types for permission classification."""
    READ = "read"
    WRITE = "write"
    DDL = "ddl"
    MIXED = "mixed"


@dataclass
class PermissionResult:
    """Result of permission extraction from a SQL statement."""
    statement_type: str
    operation_type: str
    permissions: Dict[str, Any]
    tables: List[str]
    table_aliases: Dict[str, str]
    columns: Dict[str, List[str]]
    parse_success: bool = True
    parse_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def get_read_tables(self) -> List[str]:
        """Get tables requiring read permission."""
        return list(self.permissions.get('read', {}).keys())
    
    def get_write_tables(self) -> List[str]:
        """Get tables requiring write permission."""
        return list(self.permissions.get('write', {}).keys())
    
    def get_all_required_columns(self, table: str) -> Dict[str, List[str]]:
        """Get all required columns for a table by permission type."""
        result = {}
        for perm_type in ['read', 'write']:
            if table in self.permissions.get(perm_type, {}):
                result[perm_type] = self.permissions[perm_type][table]
        return result


class SQLPermissionExtractor:
    """
    Extract permission requirements from SQL statements using sqlglot.
    
    Example:
        extractor = SQLPermissionExtractor(dialect="postgres")
        result = extractor.extract("SELECT name FROM users WHERE id = 1")
        print(result.permissions)  # {'read': {'users': ['name', 'id']}}
    """
    
    # Mapping from sqlglot expression types to (statement_type, operation_type)
    STATEMENT_TYPE_MAP = {
        exp.Select: ('SELECT', OperationType.READ),
        exp.Update: ('UPDATE', OperationType.WRITE),
        exp.Insert: ('INSERT', OperationType.WRITE),
        exp.Delete: ('DELETE', OperationType.WRITE),
        exp.Create: ('CREATE', OperationType.DDL),
        exp.Drop: ('DROP', OperationType.DDL),
        exp.Alter: ('ALTER', OperationType.DDL),
        exp.Merge: ('MERGE', OperationType.WRITE),
        exp.Union: ('UNION', OperationType.READ),
    }
    
    def __init__(self, dialect: str = "postgres"):
        """
        Initialize the extractor.
        
        Args:
            dialect: SQL dialect for parsing (postgres, sqlite, mysql, etc.)
        """
        self.dialect = dialect
    
    def extract(self, sql: str) -> PermissionResult:
        """
        Extract permission requirements from a SQL statement.
        
        Args:
            sql: SQL statement to analyze
            
        Returns:
            PermissionResult with detailed permission information
        """
        result = PermissionResult(
            statement_type='UNKNOWN',
            operation_type='read',
            permissions={
                'read': {},
                'write': {},
                'insert': [],
                'delete': [],
                'ddl': []
            },
            tables=[],
            table_aliases={},
            columns={},
            parse_success=True,
            parse_notes=[]
        )
        
        try:
            parsed = sqlglot.parse_one(sql, read=self.dialect)
            
            # 1. Identify statement type
            result.statement_type, op_type = self._get_statement_type(parsed)
            result.operation_type = op_type.value
            
            # 2. Extract tables and aliases
            result.tables, result.table_aliases = self._extract_tables(parsed)
            
            # 3. Extract columns
            result.columns = self._extract_columns(parsed, result.table_aliases)
            
            # 4. Assign permissions based on statement type
            self._assign_permissions(parsed, result)
            
        except Exception as e:
            result.parse_success = False
            result.parse_notes.append(f"Parse error: {str(e)[:200]}")
        
        return result
    
    def _get_statement_type(self, parsed) -> tuple:
        """Determine statement type and operation type."""
        for exp_type, (name, op_type) in self.STATEMENT_TYPE_MAP.items():
            if isinstance(parsed, exp_type):
                return name, op_type
        return type(parsed).__name__, OperationType.READ
    
    def _extract_tables(self, parsed) -> tuple:
        """Extract all tables and their aliases."""
        tables = set()
        aliases = {}
        
        for table in parsed.find_all(exp.Table):
            table_name = table.name
            tables.add(table_name)
            
            if table.alias:
                aliases[table.alias] = table_name
        
        return sorted(list(tables)), aliases
    
    def _extract_columns(self, parsed, aliases: Dict[str, str]) -> Dict[str, List[str]]:
        """Extract all column references grouped by table."""
        columns_by_table = defaultdict(set)
        
        for col in parsed.find_all(exp.Column):
            col_name = col.name
            table_ref = col.table if col.table else '_unbound'
            
            # Resolve alias to actual table name
            real_table = aliases.get(table_ref, table_ref)
            columns_by_table[real_table].add(col_name)
        
        return {k: sorted(list(v)) for k, v in columns_by_table.items()}
    
    def _assign_permissions(self, parsed, result: PermissionResult):
        """Assign permissions based on statement type."""
        if result.operation_type == 'read':
            self._assign_read_permissions(parsed, result)
        elif isinstance(parsed, exp.Update):
            self._assign_update_permissions(parsed, result)
        elif isinstance(parsed, exp.Insert):
            self._assign_insert_permissions(parsed, result)
        elif isinstance(parsed, exp.Delete):
            self._assign_delete_permissions(parsed, result)
        elif isinstance(parsed, exp.Create):
            self._assign_create_permissions(parsed, result)
        elif isinstance(parsed, (exp.Drop, exp.Alter)):
            self._assign_ddl_permissions(parsed, result)
    
    def _assign_read_permissions(self, parsed, result: PermissionResult):
        """Assign READ permissions for SELECT/UNION statements."""
        # All columns are READ
        for table, cols in result.columns.items():
            if table == '_unbound':
                # Distribute unbound columns to all tables
                for t in result.tables:
                    if t not in result.permissions['read']:
                        result.permissions['read'][t] = []
                    result.permissions['read'][t].extend(cols)
            else:
                result.permissions['read'][table] = cols
        
        # Ensure all tables have an entry
        for table in result.tables:
            if table not in result.permissions['read']:
                result.permissions['read'][table] = []
    
    def _assign_update_permissions(self, parsed, result: PermissionResult):
        """Assign permissions for UPDATE statements.
        
        WRITE: Only columns in SET clause (being modified)
        READ: Columns in WHERE, RETURNING, and other clauses (being read)
        """
        target_table = parsed.this.name if parsed.this else '_unknown'
        
        # SET columns -> WRITE
        # In sqlglot, UPDATE.expressions contains the SET clause assignments
        write_cols = set()
        
        if hasattr(parsed, 'expressions') and parsed.expressions:
            for expr in parsed.expressions:
                # Each expression in SET is an EQ (column = value)
                if isinstance(expr, exp.EQ):
                    col = expr.this
                    if isinstance(col, exp.Column):
                        write_cols.add(col.name)
        
        result.permissions['write'][target_table] = sorted(list(write_cols))
        
        # All other columns (WHERE, RETURNING, subqueries, etc.) -> READ
        all_cols = set()
        for cols in result.columns.values():
            all_cols.update(cols)
        read_cols = all_cols - write_cols
        result.permissions['read'][target_table] = sorted(list(read_cols))
    
    def _assign_insert_permissions(self, parsed, result: PermissionResult):
        """Assign permissions for INSERT statements.
        
        INSERT only requires 'insert' permission on the target table.
        No 'write' permission needed - 'write' is specifically for UPDATE.
        Source tables (if INSERT...SELECT) require 'read' permission.
        """
        # Extract target table
        target_table = '_unknown'
        if parsed.this:
            schema_expr = parsed.this
            if hasattr(schema_expr, 'this') and schema_expr.this:
                target_table = schema_expr.this.name
        
        result.permissions['insert'].append(target_table)
        
        # NOTE: INSERT does NOT require 'write' permission.
        # 'write' is specifically for UPDATE statements (column-level modification).
        # INSERT uses table-level 'insert' permission instead.
        
        # Source tables (from SELECT subquery) -> READ
        source_tables = [t for t in result.tables if t != target_table]
        for table in source_tables:
            cols = result.columns.get(table, [])
            # Also check aliases
            for alias, real in result.table_aliases.items():
                if real == table:
                    cols = list(set(cols + result.columns.get(alias, [])))
            result.permissions['read'][table] = cols
    
    def _assign_delete_permissions(self, parsed, result: PermissionResult):
        """Assign permissions for DELETE statements."""
        target_table = parsed.this.name if parsed.this else '_unknown'
        result.permissions['delete'].append(target_table)
        
        # WHERE columns -> READ
        all_cols = []
        for cols in result.columns.values():
            all_cols.extend(cols)
        result.permissions['read'][target_table] = list(set(all_cols))
    
    def _assign_create_permissions(self, parsed, result: PermissionResult):
        """Assign permissions for CREATE statements."""
        # Extract object being created
        if parsed.this:
            if hasattr(parsed.this, 'this'):
                obj_name = str(parsed.this.this).strip('"')
            else:
                obj_name = str(parsed.this).strip('"')
            result.permissions['ddl'].append(obj_name)
        
        # For CREATE VIEW/TABLE AS SELECT, extract referenced tables
        for select in parsed.find_all(exp.Select):
            for table in select.find_all(exp.Table):
                if table.name not in result.permissions['read']:
                    result.permissions['read'][table.name] = []
            for col in select.find_all(exp.Column):
                table_ref = col.table if col.table else '_unbound'
                real_table = result.table_aliases.get(table_ref, table_ref)
                if real_table not in result.permissions['read']:
                    result.permissions['read'][real_table] = []
                if col.name not in result.permissions['read'][real_table]:
                    result.permissions['read'][real_table].append(col.name)
    
    def _assign_ddl_permissions(self, parsed, result: PermissionResult):
        """Assign permissions for DROP/ALTER statements."""
        if parsed.this:
            obj_name = parsed.this.name if hasattr(parsed.this, 'name') else str(parsed.this)
            result.permissions['ddl'].append(obj_name.strip('"'))
    
    def extract_batch(self, sql_list: List[str]) -> List[PermissionResult]:
        """
        Extract permissions from multiple SQL statements.
        
        Args:
            sql_list: List of SQL statements
            
        Returns:
            List of PermissionResult objects
        """
        return [self.extract(sql) for sql in sql_list]
    
    def check_permission(
        self, 
        sql: str, 
        allowed_read: Dict[str, Set[str]], 
        allowed_write: Optional[Dict[str, Set[str]]] = None,
        allow_insert: Optional[Set[str]] = None,
        allow_delete: Optional[Set[str]] = None,
        allow_ddl: bool = False
    ) -> tuple:
        """
        Check if a SQL statement can be executed with given permissions.
        
        Args:
            sql: SQL statement to check
            allowed_read: {table: {columns}} for read permission
            allowed_write: {table: {columns}} for write permission
            allow_insert: Set of tables where INSERT is allowed
            allow_delete: Set of tables where DELETE is allowed
            allow_ddl: Whether DDL operations are allowed
            
        Returns:
            (allowed: bool, missing: dict) - Whether allowed and what's missing
        """
        result = self.extract(sql)
        missing = {
            'read': {},
            'write': {},
            'insert': [],
            'delete': [],
            'ddl': []
        }
        
        # Check DDL
        if result.permissions['ddl'] and not allow_ddl:
            missing['ddl'] = result.permissions['ddl']
        
        # Check INSERT
        if result.permissions['insert']:
            allow_insert = allow_insert or set()
            for table in result.permissions['insert']:
                if table not in allow_insert:
                    missing['insert'].append(table)
        
        # Check DELETE
        if result.permissions['delete']:
            allow_delete = allow_delete or set()
            for table in result.permissions['delete']:
                if table not in allow_delete:
                    missing['delete'].append(table)
        
        # Check READ permissions
        for table, cols in result.permissions['read'].items():
            if table not in allowed_read:
                missing['read'][table] = cols if cols else ['*']
            else:
                allowed_cols = allowed_read[table]
                if '*' not in allowed_cols:
                    missing_cols = [c for c in cols if c not in allowed_cols]
                    if missing_cols:
                        missing['read'][table] = missing_cols
        
        # Check WRITE permissions
        if allowed_write:
            for table, cols in result.permissions.get('write', {}).items():
                if table not in allowed_write:
                    missing['write'][table] = cols if cols else ['*']
                else:
                    allowed_cols = allowed_write[table]
                    if '*' not in allowed_cols:
                        missing_cols = [c for c in cols if c not in allowed_cols]
                        if missing_cols:
                            missing['write'][table] = missing_cols
        
        # Clean up empty entries
        missing = {k: v for k, v in missing.items() if v}
        
        return len(missing) == 0, missing


def extract_permissions(sql: str, dialect: str = "postgres") -> Dict[str, Any]:
    """
    Convenience function to extract permissions from a SQL statement.
    
    Args:
        sql: SQL statement to analyze
        dialect: SQL dialect (postgres, sqlite, mysql, etc.)
        
    Returns:
        Dictionary with permission information
    """
    extractor = SQLPermissionExtractor(dialect=dialect)
    result = extractor.extract(sql)
    return result.to_dict()


if __name__ == "__main__":
    # Test examples
    extractor = SQLPermissionExtractor(dialect="postgres")
    
    test_sqls = [
        "SELECT name, email FROM users WHERE id = 1",
        "UPDATE users SET name = 'John' WHERE id = 1",
        "INSERT INTO logs (user_id, action) SELECT id, 'login' FROM users WHERE active = true",
        "DELETE FROM sessions WHERE expires_at < NOW()",
        "CREATE VIEW active_users AS SELECT * FROM users WHERE active = true",
    ]
    
    for sql in test_sqls:
        result = extractor.extract(sql)
        print(f"\nSQL: {sql[:60]}...")
        print(f"Type: {result.statement_type}")
        print(f"Operation: {result.operation_type}")
        print(f"Permissions: {result.permissions}")
