"""
Processors module for Role-SQL-benchmark.

This module contains dataset processors for different benchmarks:
- Spider: Column-level RBAC for Spider text-to-SQL dataset
- BIRD: Column-level RBAC for BIRD text-to-SQL dataset
"""

# BIRD processors (column-level RBAC)
from .bird_describer import BirdDatabaseDescriber, DatabaseInfo, TableInfo, ColumnInfo
from .bird_role_processor import BirdRoleProcessor

# Column-level RBAC core components
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

# BIRD column-level RBAC generator
from .column_level_rbac_generator import (
    ColumnLevelRBACGenerator,
    RBACDatasetEntry,
    generate_column_level_rbac_dataset,
    DENIAL_MESSAGE
)

# Spider column-level RBAC generator (self-contained)
from .spider_column_level_rbac_generator import (
    SpiderColumnLevelRBACGenerator,
    generate_spider_column_level_dataset,
    parse_schema_sql,
    get_schema_from_db,
    # SQL parsing and difficulty computation utilities
    Schema,
    get_sql,
    compute_spider_difficulty,
)

# SQL Permission Extractor (sqlglot-based)
from .sql_permission_extractor import (
    SQLPermissionExtractor,
    PermissionResult as SQLPermissionResult,
    OperationType,
    extract_permissions,
)

# LiveSQLBench CRUD-level RBAC generator
from .livesqlbench_crud_rbac_generator import (
    LiveSQLBenchCRUDRBACGenerator,
    LiveSQLBenchCRUDEntry,
    check_crud_permission,
    generate_deterministic_crud_roles,
)

__all__ = [
    # BIRD processors
    'BirdDatabaseDescriber',
    'DatabaseInfo',
    'TableInfo', 
    'ColumnInfo',
    'BirdRoleProcessor',
    
    # Column-level RBAC core
    'SQLColumnExtractor',
    'QueryColumnInfo',
    'extract_query_columns',
    'get_required_columns_for_query',
    'ColumnPermissionChecker',
    'PermissionResult',
    'check_column_permission',
    'is_query_allowed',
    
    # BIRD column-level RBAC
    'ColumnLevelRBACGenerator',
    'RBACDatasetEntry',
    'generate_column_level_rbac_dataset',
    'DENIAL_MESSAGE',
    
    # Spider column-level RBAC
    'SpiderColumnLevelRBACGenerator',
    'generate_spider_column_level_dataset',
    'parse_schema_sql',
    'get_schema_from_db',
    'Schema',
    'get_sql',
    'compute_spider_difficulty',
    
    # SQL Permission Extractor
    'SQLPermissionExtractor',
    'SQLPermissionResult',
    'OperationType',
    'extract_permissions',
    
    # LiveSQLBench CRUD-level RBAC
    'LiveSQLBenchCRUDRBACGenerator',
    'LiveSQLBenchCRUDEntry',
    'check_crud_permission',
    'generate_deterministic_crud_roles',
]
