"""
Column-Level Role Parser Module (Simplified)

Parses LLM-generated role definitions with column-level permissions.
Supports JSON format (recommended) and Plain Text format (legacy).

Output format:
{
    "role": "FinanceAnalyst",
    "description": "Handles financial reporting",
    "columns": {
        "employees": ["id", "name"],
        "departments": ["*"]
    },
    "tables": ["employees", "departments"]
}
"""

import re
import json
import sqlite3
from typing import Dict, List, Optional


def parse_column_level_roles_json(text: str) -> List[Dict]:
    """
    Parse JSON format LLM output into structured column-level roles.
    
    Expected format:
    ```json
    [
      {"role": "...", "description": "...", "columns": {"table": ["col1", "col2"]}}
    ]
    ```
    
    Args:
        text: LLM-generated JSON text
        
    Returns:
        List of role dictionaries
    """
    # Remove markdown code blocks if present
    text = re.sub(r'^```json\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    try:
        roles = json.loads(text)
        # Normalize format: add 'tables' field for compatibility
        for role in roles:
            if 'columns' in role:
                role['tables'] = list(role['columns'].keys())
                # Normalize ["*"] to ['*']
                for table, cols in role['columns'].items():
                    if cols == ["*"]:
                        role['columns'][table] = ['*']
        return roles
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return []


def parse_column_level_roles(text: str) -> List[Dict]:
    """
    Parse LLM output into structured column-level roles.
    Auto-detects format (JSON or Plain Text).
    
    Args:
        text: LLM-generated text containing role definitions
        
    Returns:
        List of role dictionaries
    """
    text = text.strip()
    
    # Try JSON format first (check for JSON array or markdown code block)
    if text.startswith('[') or text.startswith('```json') or text.startswith('```\n['):
        return parse_column_level_roles_json(text)
    
    # Fall back to Plain Text format
    return parse_column_level_roles_text(text)


def parse_column_level_roles_text(text: str) -> List[Dict]:
    """
    Parse Plain Text format LLM output (legacy).
    
    Expected format:
        ROLE: RoleName
        DESCRIPTION: one-sentence description
        COLUMNS:
          table_name: col1, col2, col3
          table_name: *
    """
    roles = []
    current_role = None
    in_columns = False
    
    for line in text.strip().split('\n'):
        line = line.rstrip()
        
        if line.startswith('ROLE:'):
            if current_role and current_role['columns']:
                current_role['tables'] = list(current_role['columns'].keys())
                roles.append(current_role)
            current_role = {
                'role': line.split(':', 1)[1].strip(),
                'description': '',
                'columns': {}
            }
            in_columns = False
            
        elif line.startswith('DESCRIPTION:') and current_role:
            current_role['description'] = line.split(':', 1)[1].strip()
            
        elif line.startswith('COLUMNS:'):
            in_columns = True
            
        elif in_columns and ':' in line and current_role:
            match = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$', line)
            if match:
                table = match.group(1)
                cols_str = match.group(2).strip()
                if cols_str == '*':
                    cols = ['*']
                else:
                    cols = [c.strip() for c in cols_str.split(',') if c.strip()]
                current_role['columns'][table] = cols
    
    if current_role and current_role['columns']:
        current_role['tables'] = list(current_role['columns'].keys())
        roles.append(current_role)
    
    return roles


def validate_roles_against_schema(
    roles: List[Dict], 
    db_path: str
) -> Dict:
    """
    Validate parsed roles against actual database schema.
    
    Args:
        roles: List of parsed role dictionaries
        db_path: Path to SQLite database
        
    Returns:
        Validation result with details
    """
    # Load schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = {row[0].lower(): row[0] for row in cursor.fetchall()}
    
    schema = {}
    for table_lower, table_name in tables.items():
        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        schema[table_lower] = {row[1].lower(): row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    # Validate each role
    results = []
    for role in roles:
        role_result = {
            'role': role['role'],
            'valid': True,
            'invalid_tables': [],
            'invalid_columns': {}
        }
        
        for table, columns in role['columns'].items():
            table_lower = table.lower()
            
            # Check table exists
            if table_lower not in schema:
                role_result['valid'] = False
                role_result['invalid_tables'].append(table)
                continue
            
            # Check columns (skip if '*')
            if columns == ['*']:
                continue
                
            invalid_cols = []
            for col in columns:
                if col.lower() not in schema[table_lower]:
                    invalid_cols.append(col)
            
            if invalid_cols:
                role_result['valid'] = False
                role_result['invalid_columns'][table] = invalid_cols
        
        results.append(role_result)
    
    all_valid = all(r['valid'] for r in results)
    
    return {
        'all_valid': all_valid,
        'total_roles': len(roles),
        'valid_roles': sum(1 for r in results if r['valid']),
        'details': results
    }


# Convenience function for backward compatibility
def column_roles_to_table_roles(column_roles: List[Dict]) -> List[Dict]:
    """Convert column-level roles to table-level format."""
    return [
        {
            "role": r["role"],
            "description": r["description"],
            "tables": r.get("tables", list(r["columns"].keys()))
        }
        for r in column_roles
    ]
