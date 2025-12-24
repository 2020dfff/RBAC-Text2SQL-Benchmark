"""Role parser module initialization."""

from .parser import RoleParser
from .generator import RoleGenerator, ParallelRoleGenerator

# Column-level RBAC support (simplified)
from .column_level_parser import (
    parse_column_level_roles,
    parse_column_level_roles_json,
    parse_column_level_roles_text,
    validate_roles_against_schema,
    column_roles_to_table_roles
)

__all__ = [
    # Table-level (original)
    'RoleParser', 
    'RoleGenerator', 
    'ParallelRoleGenerator',
    # Column-level (simplified)
    'parse_column_level_roles',
    'parse_column_level_roles_json',
    'parse_column_level_roles_text',
    'validate_roles_against_schema',
    'column_roles_to_table_roles',
]
