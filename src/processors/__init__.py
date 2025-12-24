"""
Processors module for Role-SQL-benchmark.

This module contains dataset processors for different benchmarks:
- Spider: Original Spider text-to-SQL dataset
- BIRD: BIRD text-to-SQL dataset with evidence
"""

# Table-level processors (original)
from .spider_role_sql_generator import RoleSQLGenerator
from .bird_role_sql_generator import BirdRoleSQLGenerator
from .bird_describer import BirdDatabaseDescriber, DatabaseInfo, TableInfo, ColumnInfo
from .bird_role_processor import BirdRoleProcessor

# Column-level processors (new)
from .column_extractor import (
    SQLColumnExtractor,
    QueryColumnInfo,
    extract_query_columns,
    get_required_columns_for_query
)
from .column_permission_checker import (
    ColumnPermissionChecker,
    PermissionResult,
    check_column_permission,
    is_query_allowed
)
from .column_level_rbac_generator import (
    ColumnLevelRBACGenerator,
    RBACDatasetEntry,
    generate_column_level_rbac_dataset,
    DENIAL_MESSAGE
)

__all__ = [
    # Table-level (original)
    'RoleSQLGenerator',
    'BirdRoleSQLGenerator',
    'BirdDatabaseDescriber',
    'DatabaseInfo',
    'TableInfo', 
    'ColumnInfo',
    'BirdRoleProcessor',
    
    # Column-level (new)
    'SQLColumnExtractor',
    'QueryColumnInfo',
    'extract_query_columns',
    'get_required_columns_for_query',
    'ColumnPermissionChecker',
    'PermissionResult',
    'check_column_permission',
    'is_query_allowed',
    'ColumnLevelRBACGenerator',
    'RBACDatasetEntry',
    'generate_column_level_rbac_dataset',
    'DENIAL_MESSAGE',
]
