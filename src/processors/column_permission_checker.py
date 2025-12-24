"""
Column-Level Permission Checker for RBAC.

Checks whether a role's column-level policy allows executing a SQL query.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .column_extractor import SQLColumnExtractor, QueryColumnInfo, extract_query_columns


@dataclass
class PermissionResult:
    """Result of permission check."""
    allowed: bool
    reason: str
    required_columns: Dict[str, Set[str]]  # Columns needed by query
    granted_columns: Dict[str, Set[str]]   # Columns allowed by policy
    missing_columns: Dict[str, Set[str]]   # Columns needed but not granted
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "required_columns": {t: sorted(list(c)) for t, c in self.required_columns.items()},
            "granted_columns": {t: sorted(list(c)) for t, c in self.granted_columns.items()},
            "missing_columns": {t: sorted(list(c)) for t, c in self.missing_columns.items() if c}
        }


class ColumnPermissionChecker:
    """Check column-level permissions for SQL queries."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize checker.
        
        Args:
            db_path: Path to database for schema lookup
        """
        self.extractor = SQLColumnExtractor(db_path)
        self.db_path = db_path
    
    def normalize_policy(self, policy: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """
        Normalize policy to lowercase table names and column sets.
        
        Args:
            policy: {table_name: [col1, col2, ...]}
        
        Returns:
            Normalized policy with lowercase table names and column sets
        """
        normalized = {}
        for table, columns in policy.items():
            table_lower = table.lower()
            if columns == ["*"] or columns == "*":
                # Special case: all columns allowed
                normalized[table_lower] = {"*"}
            else:
                normalized[table_lower] = set(columns)
        return normalized
    
    def check_permission(
        self,
        sql: str,
        policy: Dict[str, List[str]],
        db_path: Optional[str] = None
    ) -> PermissionResult:
        """
        Check if policy allows executing the SQL query.
        
        Args:
            sql: SQL query to check
            policy: Column-level policy {table: [columns]}
            db_path: Optional database path for schema lookup
        
        Returns:
            PermissionResult indicating whether query is allowed
        """
        db_path = db_path or self.db_path
        
        # Extract columns required by query
        query_info = self.extractor.extract_columns(sql, db_path)
        required_columns = query_info.columns
        
        # Normalize policy
        granted_columns = self.normalize_policy(policy)
        
        # Track missing columns
        missing_columns: Dict[str, Set[str]] = {}
        
        # Check each table's columns
        for table, required_cols in required_columns.items():
            table_lower = table.lower()
            
            # Check if table is in policy
            if table_lower not in granted_columns:
                missing_columns[table] = required_cols
                continue
            
            granted = granted_columns[table_lower]
            
            # If policy grants all columns for this table
            if "*" in granted:
                continue
            
            # Check each required column
            missing_for_table = set()
            for col in required_cols:
                # Case-insensitive column match
                col_lower = col.lower()
                granted_lower = {c.lower() for c in granted}
                
                if col_lower not in granted_lower and col not in granted:
                    missing_for_table.add(col)
            
            if missing_for_table:
                missing_columns[table] = missing_for_table
        
        # Determine result
        has_missing = any(cols for cols in missing_columns.values())
        
        if has_missing:
            # Format missing columns for message
            missing_desc = []
            for table, cols in missing_columns.items():
                if cols:
                    missing_desc.append(f"{table}: {', '.join(sorted(cols))}")
            
            return PermissionResult(
                allowed=False,
                reason=f"Missing column permissions: {'; '.join(missing_desc)}",
                required_columns=required_columns,
                granted_columns=granted_columns,
                missing_columns=missing_columns
            )
        
        return PermissionResult(
            allowed=True,
            reason="All required columns are permitted",
            required_columns=required_columns,
            granted_columns=granted_columns,
            missing_columns={}
        )
    
    def check_permission_with_details(
        self,
        sql: str,
        policy: Dict[str, List[str]],
        db_path: Optional[str] = None
    ) -> Tuple[bool, Dict]:
        """
        Check permission and return detailed metadata.
        
        Args:
            sql: SQL query to check
            policy: Column-level policy
            db_path: Optional database path
        
        Returns:
            Tuple of (is_allowed, metadata_dict)
        """
        result = self.check_permission(sql, policy, db_path)
        
        metadata = {
            "permission": "allowed" if result.allowed else "denied",
            "reason": result.reason,
            "query_columns": result.to_dict()["required_columns"],
            "missing_columns": result.to_dict()["missing_columns"]
        }
        
        return result.allowed, metadata


def check_column_permission(
    sql: str,
    policy: Dict[str, List[str]],
    db_path: Optional[str] = None
) -> PermissionResult:
    """
    Convenience function to check column permission.
    
    Args:
        sql: SQL query
        policy: Column-level policy {table: [columns]}
        db_path: Optional database path
    
    Returns:
        PermissionResult
    """
    checker = ColumnPermissionChecker(db_path)
    return checker.check_permission(sql, policy, db_path)


def is_query_allowed(
    sql: str,
    policy: Dict[str, List[str]],
    db_path: Optional[str] = None
) -> bool:
    """
    Simple check if query is allowed.
    
    Args:
        sql: SQL query
        policy: Column-level policy
        db_path: Optional database path
    
    Returns:
        True if allowed, False otherwise
    """
    result = check_column_permission(sql, policy, db_path)
    return result.allowed
