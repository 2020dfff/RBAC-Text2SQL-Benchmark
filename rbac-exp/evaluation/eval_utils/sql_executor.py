"""
SQL Execution utilities for evaluation.
Supports SQLite (Spider/Bird) and PostgreSQL (LiveSQLBench).
"""
import os
import sqlite3
import logging
from typing import Optional, Tuple, Any, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SQLExecutionError(Exception):
    """Exception for SQL execution errors."""
    pass


@contextmanager
def sqlite_connection(db_path: str):
    """Context manager for SQLite connection."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        yield conn
    finally:
        conn.close()


def execute_sqlite(
    sql: str,
    db_path: str,
    timeout: int = 30,
) -> Tuple[bool, Optional[List[Any]], Optional[str]]:
    """
    Execute SQL on SQLite database.
    
    Args:
        sql: SQL query to execute
        db_path: Path to SQLite database file
        timeout: Query timeout in seconds
        
    Returns:
        Tuple of (success, results, error_message)
    """
    if not os.path.exists(db_path):
        return False, None, f"Database not found: {db_path}"
    
    try:
        with sqlite_connection(db_path) as conn:
            conn.execute("PRAGMA read_uncommitted = true")
            cursor = conn.cursor()
            
            # Execute with timeout (SQLite doesn't support server-side timeout)
            cursor.execute(sql)
            results = cursor.fetchall()
            
            return True, results, None
            
    except sqlite3.Error as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, str(e)


def try_execute_postgresql(
    sql: str,
    db_name: str,
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: str = "",
    timeout: int = 30,
) -> Tuple[bool, Optional[List[Any]], Optional[str]]:
    """
    Execute SQL on PostgreSQL database.
    
    Args:
        sql: SQL query to execute
        db_name: Database name
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        timeout: Query timeout in seconds
        
    Returns:
        Tuple of (success, results, error_message)
    """
    try:
        import psycopg2
        from psycopg2 import sql as psql
    except ImportError:
        return False, None, "psycopg2 not installed. Install with: pip install psycopg2-binary"
    
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=db_name,
            user=user,
            password=password,
            connect_timeout=timeout,
        )
        
        try:
            cursor = conn.cursor()
            cursor.execute(f"SET statement_timeout = '{timeout}s'")
            cursor.execute(sql)
            results = cursor.fetchall()
            return True, results, None
        finally:
            conn.close()
            
    except Exception as e:
        return False, None, str(e)


def normalize_results(results: List[Any]) -> List[Tuple]:
    """Normalize query results for comparison."""
    if not results:
        return []
    
    normalized = []
    for row in results:
        # Convert to tuple and normalize values
        norm_row = tuple(
            str(v).lower().strip() if isinstance(v, str) else v
            for v in row
        )
        normalized.append(norm_row)
    
    return sorted(normalized)


def compare_results(
    pred_results: List[Any],
    gold_results: List[Any],
    ignore_order: bool = True,
) -> bool:
    """
    Compare predicted and gold results.
    
    Args:
        pred_results: Predicted query results
        gold_results: Gold query results
        ignore_order: Whether to ignore row order
        
    Returns:
        True if results match
    """
    if pred_results is None or gold_results is None:
        return pred_results is None and gold_results is None
    
    pred_norm = normalize_results(pred_results)
    gold_norm = normalize_results(gold_results)
    
    if ignore_order:
        return set(map(tuple, pred_norm)) == set(map(tuple, gold_norm))
    else:
        return pred_norm == gold_norm


class SQLExecutor:
    """SQL executor supporting multiple database types."""
    
    def __init__(
        self,
        db_type: str = "sqlite",
        db_root_path: Optional[str] = None,
        pg_host: str = "localhost",
        pg_port: int = 5432,
        pg_user: str = "postgres",
        pg_password: str = "",
        timeout: int = 30,
    ):
        self.db_type = db_type
        self.db_root_path = db_root_path
        self.pg_host = pg_host
        self.pg_port = pg_port
        self.pg_user = pg_user
        self.pg_password = pg_password
        self.timeout = timeout
    
    def get_db_path(self, db_name: str) -> str:
        """Get full database path for SQLite databases."""
        if self.db_root_path:
            return os.path.join(self.db_root_path, db_name, f"{db_name}.sqlite")
        return db_name
    
    def execute(
        self,
        sql: str,
        db_name: str,
    ) -> Tuple[bool, Optional[List[Any]], Optional[str]]:
        """
        Execute SQL query.
        
        Args:
            sql: SQL query to execute
            db_name: Database name
            
        Returns:
            Tuple of (success, results, error_message)
        """
        if not sql or not sql.strip():
            return False, None, "Empty SQL query"
        
        if self.db_type == "sqlite":
            db_path = self.get_db_path(db_name)
            return execute_sqlite(sql, db_path, self.timeout)
        elif self.db_type == "postgresql":
            return try_execute_postgresql(
                sql, db_name,
                self.pg_host, self.pg_port,
                self.pg_user, self.pg_password,
                self.timeout,
            )
        else:
            return False, None, f"Unsupported database type: {self.db_type}"
    
    def verify_sql_correctness(
        self,
        pred_sql: str,
        gold_sql: str,
        db_name: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify if predicted SQL produces same results as gold SQL.
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold SQL query
            db_name: Database name
            
        Returns:
            Tuple of (is_correct, error_message)
        """
        # Execute gold SQL
        gold_success, gold_results, gold_error = self.execute(gold_sql, db_name)
        if not gold_success:
            return False, f"Gold SQL execution failed: {gold_error}"
        
        # Execute predicted SQL
        pred_success, pred_results, pred_error = self.execute(pred_sql, db_name)
        if not pred_success:
            return False, f"Predicted SQL execution failed: {pred_error}"
        
        # Compare results
        is_correct = compare_results(pred_results, gold_results)
        
        return is_correct, None
