"""
SQL Column Extractor for Column-Level RBAC.

Extracts columns referenced in SQL queries for permission checking.
Handles CTEs (WITH clauses) by filtering out derived table/column names.
"""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

try:
    from sql_metadata import Parser
    HAS_SQL_METADATA = True
except ImportError:
    HAS_SQL_METADATA = False

import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Parenthesis
from sqlparse.tokens import Keyword, DML


@dataclass
class QueryColumnInfo:
    """Information about columns used in a SQL query."""
    tables: Set[str]
    columns: Dict[str, Set[str]]  # {table_name: {col1, col2, ...}}
    has_star: bool  # Whether query uses SELECT *
    star_tables: Set[str]  # Tables that use * (need expansion)
    raw_columns: List[str]  # Raw column references before resolution
    cte_names: Set[str]  # CTE (WITH clause) table names to exclude


class SQLColumnExtractor:
    """Extract columns from SQL queries."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize extractor.
        
        Args:
            db_path: Path to SQLite database for schema lookup (for SELECT * expansion)
        """
        self.db_path = db_path
        self._schema_cache: Dict[str, Dict[str, List[str]]] = {}
    
    def _extract_cte_names(self, sql: str) -> Set[str]:
        """
        Extract CTE (Common Table Expression) names from WITH clause.
        
        CTEs define virtual tables that should NOT be checked against
        the database schema for permission validation.
        
        Example:
            WITH sales_summary AS (...), monthly_totals AS (...)
            SELECT * FROM sales_summary
        
        Returns: {'sales_summary', 'monthly_totals'}
        """
        cte_names = set()
        
        # Match: WITH cte_name AS (...), cte_name2 AS (...)
        # Handle both simple and recursive CTEs
        sql_upper = sql.upper()
        if 'WITH' not in sql_upper:
            return cte_names
        
        # Find WITH clause and extract CTE names
        # Pattern: WITH [RECURSIVE] name AS (...) [, name2 AS (...)]
        with_pattern = re.compile(
            r'\bWITH\s+(?:RECURSIVE\s+)?'
            r'(\w+)\s+AS\s*\(',
            re.IGNORECASE
        )
        
        # Also find additional CTEs after comma: , name AS (
        additional_cte_pattern = re.compile(
            r',\s*(\w+)\s+AS\s*\(',
            re.IGNORECASE
        )
        
        # Extract first CTE name
        match = with_pattern.search(sql)
        if match:
            cte_names.add(match.group(1).lower())
        
        # Extract additional CTE names
        for match in additional_cte_pattern.finditer(sql):
            cte_names.add(match.group(1).lower())
        
        return cte_names
    
    def get_db_schema(self, db_path: str) -> Dict[str, List[str]]:
        """Get schema from database: {table_name: [col1, col2, ...]}"""
        if db_path in self._schema_cache:
            return self._schema_cache[db_path]
        
        schema = {}
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                cursor.execute(f"PRAGMA table_info(`{table}`)")
                columns = [row[1] for row in cursor.fetchall()]
                schema[table.lower()] = columns
            
            conn.close()
            self._schema_cache[db_path] = schema
        except Exception as e:
            print(f"Warning: Could not read schema from {db_path}: {e}")
        
        return schema
    
    def extract_columns(self, sql: str, db_path: Optional[str] = None) -> QueryColumnInfo:
        """
        Extract columns from SQL query.
        
        Args:
            sql: SQL query string
            db_path: Optional database path for SELECT * expansion
        
        Returns:
            QueryColumnInfo with extracted column information
        """
        db_path = db_path or self.db_path
        
        # Try sql_metadata first (more accurate)
        if HAS_SQL_METADATA:
            try:
                return self._extract_with_sql_metadata(sql, db_path)
            except Exception:
                pass
        
        # Fallback to sqlparse
        return self._extract_with_sqlparse(sql, db_path)
    
    def _extract_with_sql_metadata(self, sql: str, db_path: Optional[str]) -> QueryColumnInfo:
        """Extract using sql_metadata library."""
        parser = Parser(sql)
        
        # Extract CTE names to filter out
        cte_names = self._extract_cte_names(sql)
        
        # Get schema for validation
        schema = self.get_db_schema(db_path) if db_path else {}
        
        # Filter out CTE names and non-existent tables (sql_metadata may misparse CASE, etc.)
        raw_tables = set(t.lower() for t in parser.tables if t.lower() not in cte_names)
        
        # Only keep tables that exist in schema (if schema is available)
        if schema:
            tables = set(t for t in raw_tables if t in schema)
        else:
            tables = raw_tables
        
        raw_columns = parser.columns
        
        # Check for SELECT *
        has_star = '*' in raw_columns or any('*' in str(c) for c in raw_columns)
        star_tables = set()
        
        # Build column mapping (only for real tables)
        columns: Dict[str, Set[str]] = {t: set() for t in tables}
        
        for col in raw_columns:
            col_str = str(col).strip()
            
            # Skip aggregate functions without column
            if col_str in ('*', 'COUNT(*)', 'count(*)'):
                if tables:
                    star_tables = tables.copy()
                continue
            
            # Handle table.column format
            if '.' in col_str:
                parts = col_str.split('.')
                table_name = parts[0].strip('`"[] ').lower()
                col_name = parts[1].strip('`"[] ')
                
                # Skip if table is a CTE
                if table_name in cte_names:
                    continue
                
                if col_name == '*':
                    star_tables.add(table_name)
                elif table_name in columns:
                    columns[table_name].add(col_name)
            else:
                # Column without table prefix - try to resolve against real tables only
                col_name = col_str.strip('`"[] ')
                resolved = self._resolve_column_table(col_name, tables, schema)
                if resolved:
                    table_name, actual_col = resolved
                    columns[table_name].add(actual_col)
                # Note: If column can't be resolved to a real table, it might be
                # a CTE-derived column (e.g., aliases defined in CTE SELECT).
                # We skip these instead of adding to first table.
        
        # Expand SELECT * if needed (only for real tables)
        if star_tables and schema:
            for table in star_tables:
                if table in schema and table not in cte_names:
                    columns[table] = set(schema[table])
        
        return QueryColumnInfo(
            tables=tables,
            columns=columns,
            has_star=has_star,
            star_tables=star_tables,
            raw_columns=raw_columns,
            cte_names=cte_names
        )
    
    def _extract_with_sqlparse(self, sql: str, db_path: Optional[str]) -> QueryColumnInfo:
        """Extract using sqlparse (fallback)."""
        parsed = sqlparse.parse(sql)
        if not parsed:
            return QueryColumnInfo(set(), {}, False, set(), [], set())
        
        # Extract CTE names first
        cte_names = self._extract_cte_names(sql)
        
        stmt = parsed[0]
        raw_tables = self._extract_tables_sqlparse(stmt)
        
        # Filter out CTE names from tables
        raw_tables = raw_tables - cte_names
        
        # Get schema for validation
        schema = self.get_db_schema(db_path) if db_path else {}
        
        # Filter to only keep tables that exist in schema (if schema available)
        # This filters out false positives like CTE column aliases misidentified as tables
        if schema:
            tables = set(t for t in raw_tables if t in schema)
        else:
            tables = raw_tables
        
        columns: Dict[str, Set[str]] = {t: set() for t in tables}
        raw_columns = []
        has_star = False
        star_tables = set()
        
        # Extract columns from SELECT clause
        select_seen = False
        for token in stmt.tokens:
            if token.ttype is DML and token.value.upper() == 'SELECT':
                select_seen = True
                continue
            
            if select_seen:
                if token.ttype is Keyword:
                    break
                
                cols = self._extract_columns_from_token(token)
                for col in cols:
                    raw_columns.append(col)
                    
                    if col == '*':
                        has_star = True
                        star_tables = tables.copy()
                        continue
                    
                    if '.' in col:
                        parts = col.split('.')
                        table = parts[0].strip('`"[] ').lower()
                        col_name = parts[1].strip('`"[] ')
                        if col_name == '*':
                            star_tables.add(table)
                        elif table in columns:
                            columns[table].add(col_name)
                    else:
                        col_name = col.strip('`"[] ')
                        resolved = self._resolve_column_table(col_name, tables, schema)
                        if resolved:
                            columns[resolved[0]].add(resolved[1])
                        elif tables:
                            columns[next(iter(tables))].add(col_name)
        
        # Also extract from WHERE, GROUP BY, ORDER BY
        where_cols = self._extract_where_columns(sql)
        for col in where_cols:
            raw_columns.append(col)
            if '.' in col:
                parts = col.split('.')
                table = parts[0].strip('`"[] ').lower()
                col_name = parts[1].strip('`"[] ')
                # Skip CTE references
                if table in cte_names:
                    continue
                if table in columns:
                    columns[table].add(col_name)
            else:
                col_name = col.strip('`"[] ')
                resolved = self._resolve_column_table(col_name, tables, schema)
                if resolved:
                    columns[resolved[0]].add(resolved[1])
        
        # Expand SELECT * (only for real tables)
        if star_tables and schema:
            for table in star_tables:
                if table in schema and table not in cte_names:
                    columns[table] = set(schema[table])
        
        return QueryColumnInfo(
            tables=tables,
            columns=columns,
            has_star=has_star,
            star_tables=star_tables,
            raw_columns=raw_columns,
            cte_names=cte_names
        )
    
    def _extract_tables_sqlparse(self, stmt) -> Set[str]:
        """Extract table names from parsed statement."""
        tables = set()
        from_seen = False
        
        for token in stmt.tokens:
            if token.ttype is Keyword:
                if token.value.upper() in ('FROM', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 
                                           'RIGHT JOIN', 'OUTER JOIN', 'CROSS JOIN'):
                    from_seen = True
                elif token.value.upper() in ('WHERE', 'GROUP', 'ORDER', 'LIMIT', 'HAVING'):
                    from_seen = False
            elif from_seen:
                if isinstance(token, Identifier):
                    tables.add(token.get_real_name().lower())
                elif isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        if isinstance(identifier, Identifier):
                            tables.add(identifier.get_real_name().lower())
        
        # Also try regex for simple cases
        from_match = re.findall(r'\bFROM\s+[`"\[]?(\w+)[`"\]]?', str(stmt), re.IGNORECASE)
        join_match = re.findall(r'\bJOIN\s+[`"\[]?(\w+)[`"\]]?', str(stmt), re.IGNORECASE)
        
        for t in from_match + join_match:
            tables.add(t.lower())
        
        return tables
    
    def _extract_columns_from_token(self, token) -> List[str]:
        """Extract column names from a token."""
        columns = []
        
        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                cols = self._extract_columns_from_token(identifier)
                columns.extend(cols)
        elif isinstance(token, Identifier):
            name = token.get_real_name()
            if name:
                columns.append(name)
        elif hasattr(token, 'value'):
            val = str(token.value).strip()
            if val and not val.upper().startswith(('SELECT', 'FROM', 'WHERE')):
                # Try to extract column references
                col_matches = re.findall(r'[`"\[]?(\w+(?:\.\w+)?)[`"\]]?', val)
                columns.extend(col_matches)
        
        return columns
    
    def _extract_where_columns(self, sql: str) -> List[str]:
        """Extract column references from WHERE, GROUP BY, ORDER BY clauses."""
        columns = []
        
        # Simple regex extraction for column references
        # Matches: column_name, table.column, `column name`, etc.
        patterns = [
            r'WHERE\s+.*?(?=GROUP|ORDER|LIMIT|HAVING|$)',
            r'GROUP\s+BY\s+.*?(?=ORDER|LIMIT|HAVING|$)',
            r'ORDER\s+BY\s+.*?(?=LIMIT|$)',
            r'HAVING\s+.*?(?=ORDER|LIMIT|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
            if match:
                clause = match.group(0)
                # Extract backtick-quoted columns
                backtick_cols = re.findall(r'`([^`]+)`', clause)
                columns.extend(backtick_cols)
                
                # Extract table.column references
                table_cols = re.findall(r'(\w+)\.(`[^`]+`|\w+)', clause)
                for table, col in table_cols:
                    columns.append(f"{table}.{col.strip('`')}")
        
        return columns
    
    def _resolve_column_table(
        self, 
        column: str, 
        tables: Set[str], 
        schema: Dict[str, List[str]]
    ) -> Optional[Tuple[str, str]]:
        """Resolve which table a column belongs to."""
        column_lower = column.lower()
        
        for table in tables:
            if table in schema:
                # Case-insensitive column match
                for schema_col in schema[table]:
                    if schema_col.lower() == column_lower:
                        return (table, schema_col)
        
        return None


def extract_query_columns(sql: str, db_path: Optional[str] = None) -> QueryColumnInfo:
    """
    Convenience function to extract columns from SQL.
    
    Args:
        sql: SQL query string
        db_path: Optional path to database for schema lookup
    
    Returns:
        QueryColumnInfo with extracted information
    """
    extractor = SQLColumnExtractor(db_path)
    return extractor.extract_columns(sql, db_path)


def get_required_columns_for_query(sql: str, db_path: str) -> Dict[str, Set[str]]:
    """
    Get the columns required to execute a query.
    
    Args:
        sql: SQL query string
        db_path: Path to database
    
    Returns:
        Dict mapping table names to sets of required column names
    """
    info = extract_query_columns(sql, db_path)
    return info.columns
