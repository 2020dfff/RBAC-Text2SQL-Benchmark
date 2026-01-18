#!/usr/bin/env python3
"""
Baseline Text2SQL Evaluation for LiveSQLBench-Full (PostgreSQL Backend)

This module evaluates pure Text2SQL accuracy without RBAC constraints.
Uses PostgreSQL for query execution (same as RBAC evaluation).

For LiveSQLBench baseline:
- No role/permission checking
- Uses SystemManager role items (always allowed, has valid gold_sql)
- Evaluates execution accuracy only
- Uses ephemeral databases for management operations to prevent pollution

Usage:
    python -m rbac-exp.evaluation.evaluate_crud_baseline \
        --pred rbac-exp/output/pred/pred_deepseek-chat_livesqlbench_baseline.sql \
        --db_user postgres --db_password your_password
"""

import argparse
import json
import logging
import os
import sys
import subprocess
import queue
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PostgreSQL support
try:
    import psycopg2
    from psycopg2 import OperationalError
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("WARNING: psycopg2 not installed. Install with: pip install psycopg2-binary")

from func_timeout import FunctionTimedOut, func_timeout

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIMEOUT = 60  # seconds

# Operation types for statistics
OPERATION_TYPES = ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "UNION", "Analyze", "unknown")

# Operations that modify database state (require ephemeral DB)
MODIFYING_OPERATIONS = {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"}


# ===========================================================================
# Ephemeral Database Management (from evaluate_crud_level.py)
# ===========================================================================

def check_template_exists(base_db: str, config: Dict[str, Any]) -> bool:
    """Check if template database exists."""
    template_name = f"{base_db}_template"
    try:
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            dbname="postgres"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (template_name,))
        exists = cursor.fetchone() is not None
        cursor.close()
        conn.close()
        return exists
    except Exception as e:
        logger.debug(f"Error checking template {template_name}: {e}")
        return False


def create_ephemeral_db(base_db: str, worker_id: int, config: Dict[str, Any]) -> Optional[str]:
    """Create an ephemeral database copy for a worker."""
    template_name = f"{base_db}_template"
    ephemeral_name = f"{base_db}_baseline_process_{worker_id}"
    
    if not check_template_exists(base_db, config):
        logger.error(f"Template {template_name} not found for database {base_db}")
        return None
    
    env_vars = os.environ.copy()
    env_vars["PGPASSWORD"] = config["password"]
    
    try:
        # Drop if exists
        drop_cmd = [
            "dropdb", "--if-exists",
            "-h", config["host"],
            "-p", str(config["port"]),
            "-U", config["user"],
            ephemeral_name
        ]
        subprocess.run(drop_cmd, check=False, env=env_vars, timeout=30,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Create from template
        create_cmd = [
            "createdb",
            "-h", config["host"],
            "-p", str(config["port"]),
            "-U", config["user"],
            ephemeral_name,
            "--template", template_name
        ]
        subprocess.run(create_cmd, check=True, env=env_vars, timeout=120,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        logger.debug(f"Created ephemeral database: {ephemeral_name}")
        return ephemeral_name
    except Exception as e:
        logger.error(f"Failed to create ephemeral db {ephemeral_name}: {e}")
        return None


def reset_ephemeral_database(ephemeral_db: str, config: Dict[str, Any]) -> bool:
    """Reset an ephemeral database by dropping and recreating from template."""
    base_db = ephemeral_db.rsplit('_baseline_process_', 1)[0]
    worker_id = int(ephemeral_db.rsplit('_', 1)[1])
    
    env_vars = os.environ.copy()
    env_vars["PGPASSWORD"] = config["password"]
    template_name = f"{base_db}_template"
    
    try:
        # Terminate connections
        terminate_cmd = [
            "psql",
            "-h", config["host"],
            "-p", str(config["port"]),
            "-U", config["user"],
            "-d", "postgres",
            "-c", f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{ephemeral_db}' AND pid <> pg_backend_pid();"
        ]
        subprocess.run(terminate_cmd, check=False, env=env_vars, timeout=30,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Drop
        drop_cmd = [
            "dropdb", "--if-exists",
            "-h", config["host"],
            "-p", str(config["port"]),
            "-U", config["user"],
            ephemeral_db
        ]
        subprocess.run(drop_cmd, check=True, env=env_vars, timeout=60,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Recreate from template
        create_cmd = [
            "createdb",
            "-h", config["host"],
            "-p", str(config["port"]),
            "-U", config["user"],
            ephemeral_db,
            "--template", template_name
        ]
        subprocess.run(create_cmd, check=True, env=env_vars, timeout=120,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return True
    except Exception as e:
        logger.error(f"Error resetting ephemeral db {ephemeral_db}: {e}")
        return False


def cleanup_ephemeral_databases(config: Dict[str, Any]):
    """Clean up all baseline ephemeral databases."""
    env_vars = os.environ.copy()
    env_vars["PGPASSWORD"] = config["password"]
    
    try:
        # Get list of baseline ephemeral databases
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            dbname="postgres"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT datname FROM pg_database WHERE datname LIKE '%_baseline_process_%';")
        ephemeral_dbs = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        # Drop each ephemeral database
        for db_name in ephemeral_dbs:
            drop_cmd = [
                "dropdb", "--if-exists",
                "-h", config["host"],
                "-p", str(config["port"]),
                "-U", config["user"],
                db_name
            ]
            subprocess.run(drop_cmd, check=False, env=env_vars, timeout=60,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.debug(f"Cleaned up ephemeral database: {db_name}")
        
        if ephemeral_dbs:
            logger.info(f"Cleaned up {len(ephemeral_dbs)} ephemeral databases")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


# All LiveSQLBench-Full databases
LIVESQLBENCH_DATABASES = [
    "archeology_scan",
    "cold_chain_pharma_compliance",
    "cross_border",
    "crypto_exchange",
    "cybermarket_pattern",
    "disaster_relief",
    "exchange_traded_funds",
    "fake_account",
    "households",
    "hulushows",
    "insider_trading",
    "labor_certification_applications",
    "mental_health",
    "museum_artifact",
    "organ_transplant",
    "planets_data",
    "polar_equipment",
    "reverse_logistics",
    "robot_fault_prediction",
    "solar_panel",
    "sports_events",
    "virtual_idol",
]


def reinitialize_databases(
    config: Dict[str, Any],
    dumps_dir: str = "data/livesqlbench-full-postgresql/bird-interact-full-dumps",
) -> tuple:
    """
    Reinitialize PostgreSQL databases from SQL dumps.
    
    This function:
    1. Drops and recreates all base databases from dump files
    2. Recreates template databases from base databases
    3. Ensures clean state before evaluation
    
    Args:
        config: PostgreSQL connection config
        dumps_dir: Directory containing {db_name}_template/{db_name}_full.sql files
    
    Returns:
        Tuple of (success_count, fail_count)
    """
    env_vars = os.environ.copy()
    env_vars["PGPASSWORD"] = config["password"]
    
    logger.info("=" * 60)
    logger.info("REINITIALIZING DATABASES FOR BASELINE EVALUATION")
    logger.info(f"Dumps directory: {dumps_dir}")
    logger.info(f"Databases to reinitialize: {len(LIVESQLBENCH_DATABASES)}")
    logger.info("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for db_name in LIVESQLBENCH_DATABASES:
        template_dir = os.path.join(dumps_dir, f"{db_name}_template")
        full_sql = os.path.join(template_dir, f"{db_name}_full.sql")
        
        if not os.path.exists(full_sql):
            logger.warning(f"Skip {db_name}: dump file not found at {full_sql}")
            fail_count += 1
            continue
        
        logger.info(f"Reinitializing {db_name}...")
        
        try:
            # Drop and recreate base database
            subprocess.run([
                "dropdb", "--if-exists",
                "-h", config["host"],
                "-p", str(config["port"]),
                "-U", config["user"],
                db_name
            ], check=True, env=env_vars, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            subprocess.run([
                "createdb",
                "-h", config["host"],
                "-p", str(config["port"]),
                "-U", config["user"],
                db_name
            ], check=True, env=env_vars, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Import SQL dump
            with open(full_sql, 'r') as f:
                subprocess.run([
                    "psql",
                    "-h", config["host"],
                    "-p", str(config["port"]),
                    "-U", config["user"],
                    "-d", db_name,
                    "-q"
                ], stdin=f, check=True, env=env_vars, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Recreate template database
            subprocess.run([
                "dropdb", "--if-exists",
                "-h", config["host"],
                "-p", str(config["port"]),
                "-U", config["user"],
                f"{db_name}_template"
            ], check=True, env=env_vars, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            subprocess.run([
                "createdb",
                "-h", config["host"],
                "-p", str(config["port"]),
                "-U", config["user"],
                f"{db_name}_template",
                "--template", db_name
            ], check=True, env=env_vars, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            logger.info(f"  ✅ {db_name} reinitialized successfully")
            success_count += 1
            
        except Exception as e:
            logger.error(f"  ❌ Failed to reinitialize {db_name}: {e}")
            fail_count += 1
    
    logger.info("=" * 60)
    logger.info(f"Database reinitialization complete!")
    logger.info(f"Success: {success_count}, Failed: {fail_count}")
    logger.info("=" * 60)
    
    return success_count, fail_count


# ===========================================================================
# SQL Preprocessing Functions (from evaluate_crud_level.py)
# ===========================================================================

def remove_round_functions(sql_string: str) -> str:
    """
    Remove all ROUND() function calls from a SQL string, including nested ones.
    This implementation is from official livesqlbench/evaluation/src/test_utils.py
    """
    import re
    
    def find_matching_paren(text, start_pos):
        """Find the position of the matching closing parenthesis."""
        paren_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == '(':
                paren_count += 1
            elif text[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    return i
        return -1
    
    def find_first_arg_end(text, start_pos):
        """Find the end of the first argument, accounting for nested parentheses."""
        paren_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == '(':
                paren_count += 1
            elif text[i] == ')':
                if paren_count == 0:
                    return i  # End of ROUND function
                paren_count -= 1
            elif text[i] == ',' and paren_count == 0:
                return i  # End of first argument
        return len(text)
    
    result = sql_string
    
    while True:
        # Find ROUND function (case insensitive)
        pattern = re.compile(r'ROUND\s*\(', re.IGNORECASE)
        match = pattern.search(result)
        
        if not match:
            break
            
        start_pos = match.start()
        open_paren_pos = match.end() - 1
        
        # Find the end of the first argument
        first_arg_end = find_first_arg_end(result, open_paren_pos + 1)
        
        # Find the matching closing parenthesis
        close_paren_pos = find_matching_paren(result, open_paren_pos)
        
        if close_paren_pos == -1:
            break  # Malformed SQL, can't find closing paren
        
        # Extract the first argument
        first_arg = result[open_paren_pos + 1:first_arg_end].strip()
        
        # Replace ROUND(...) with just the first argument
        result = result[:start_pos] + first_arg + result[close_paren_pos + 1:]
    
    return result


def remove_round(sql_list: list) -> list:
    """Remove ROUND function calls from SQL queries."""
    return [remove_round_functions(sql) for sql in sql_list]


def process_decimals_recursive(item, decimal_places: int = 2):
    """Recursively process decimals in any data structure."""
    from decimal import Decimal, ROUND_HALF_UP
    quantizer = Decimal(1).scaleb(-decimal_places)
    
    if isinstance(item, Decimal):
        return item.quantize(quantizer, rounding=ROUND_HALF_UP)
    elif isinstance(item, float):
        return Decimal(str(item)).quantize(quantizer, rounding=ROUND_HALF_UP)
    elif isinstance(item, (list, tuple)):
        return type(item)(process_decimals_recursive(x, decimal_places) for x in item)
    elif isinstance(item, dict):
        return {k: process_decimals_recursive(v, decimal_places) for k, v in item.items()}
    else:
        return item


def preprocess_results(results: list, decimal_places: int = 2) -> list:
    """Process the result set for comparison."""
    processed = []
    for result in results:
        if isinstance(result, tuple):
            processed_row = tuple(
                process_decimals_recursive(val, decimal_places)
                if isinstance(val, (float, int)) or val.__class__.__name__ == 'Decimal'
                else val
                for val in result
            )
            processed.append(processed_row)
        else:
            processed.append(result)
    return processed


def get_operation_type(sql: str) -> str:
    """Determine SQL operation type from statement."""
    if not sql:
        return "unknown"
    
    sql_upper = sql.strip().upper()
    
    # Check for compound operations first
    if " UNION " in sql_upper:
        return "UNION"
    
    # Handle CTE (WITH clause) - need to find the main operation
    if sql_upper.startswith("WITH"):
        # Find the main query after CTE
        # Look for main operation keywords after the last closing parenthesis of CTE
        import re
        # Try to find SELECT/INSERT/UPDATE/DELETE after WITH ... )
        main_ops = re.search(r'\)\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)', sql_upper)
        if main_ops:
            return main_ops.group(1)
        # Default to SELECT for WITH queries if can't determine
        return "SELECT"
    
    # Handle subqueries starting with (
    if sql_upper.startswith("("):
        if "SELECT" in sql_upper[:100]:  # Check first 100 chars
            return "SELECT"
        return "unknown"
    
    # Check operation keywords
    for op in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]:
        if sql_upper.startswith(op):
            return op
    
    # Check for analysis/explain
    if sql_upper.startswith("EXPLAIN") or sql_upper.startswith("ANALYZE"):
        return "Analyze"
    
    return "unknown"


def execute_sql_postgres(
    sql: str,
    db_name: str,
    config: Dict[str, Any],
    timeout: int = TIMEOUT,
    is_modifying: bool = False,
) -> tuple:
    """
    Execute SQL on PostgreSQL and return results.
    
    For modifying operations, uses ephemeral database to prevent pollution.
    For SELECT queries, uses original database directly.
    
    Args:
        sql: SQL query to execute
        db_name: Database name (original or ephemeral)
        config: PostgreSQL connection configuration
        timeout: Query timeout in seconds
        is_modifying: Whether this is a modifying operation (INSERT/UPDATE/DELETE/CREATE/etc.)
    
    Returns:
        (success: bool, result: Any, error: Optional[str])
    """
    if not POSTGRES_AVAILABLE:
        return False, None, "psycopg2 not installed"
    
    conn = None
    try:
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            dbname=db_name,
            connect_timeout=10
        )
        conn.autocommit = True
        
        with conn.cursor() as cursor:
            cursor.execute(f"SET statement_timeout = '{timeout * 1000}';")
            cursor.execute(sql)
            
            # Try to fetch results for SELECT queries
            try:
                results = cursor.fetchall()
                return True, results, None
            except psycopg2.ProgrammingError:
                # Non-SELECT query (INSERT/UPDATE/DELETE/etc.)
                return True, [], None
                
    except psycopg2.Error as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, str(e)
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def normalize_result(result: Any) -> set:
    """Normalize SQL result for comparison."""
    if result is None:
        return set()
    
    # Preprocess to handle decimals
    if isinstance(result, (list, tuple)):
        result = preprocess_results(result)
    
    if isinstance(result, (list, tuple)):
        normalized = set()
        for row in result:
            if isinstance(row, (list, tuple)):
                # Convert each element to string for comparison
                normalized.add(tuple(str(x) if x is not None else "NULL" for x in row))
            else:
                normalized.add((str(row) if row is not None else "NULL",))
        return normalized
    
    return {(str(result),)}


def compare_results(pred_result: Any, gold_result: Any) -> bool:
    """Compare two SQL execution results."""
    pred_set = normalize_result(pred_result)
    gold_set = normalize_result(gold_result)
    return pred_set == gold_set


def evaluate_baseline_livesqlbench(
    prediction_path: str,
    pg_config: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate LiveSQLBench baseline Text2SQL predictions.
    
    Args:
        prediction_path: Path to prediction file (.sql or _detailed.json)
        pg_config: PostgreSQL connection configuration
        output_dir: Output directory for results
        
    Returns:
        Evaluation results dictionary
    """
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join("rbac-exp", "output", "eval_result")
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle both .sql and _detailed.json files
    if prediction_path.endswith('.sql'):
        detailed_path = prediction_path.replace('.sql', '_detailed.json')
    else:
        detailed_path = prediction_path
    
    if not os.path.exists(detailed_path):
        logger.error(f"Detailed prediction file not found: {detailed_path}")
        return {"error": f"File not found: {detailed_path}"}
    
    # Load predictions
    logger.info(f"Loading predictions from: {detailed_path}")
    with open(detailed_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Extract base name for output files
    base = os.path.splitext(os.path.basename(prediction_path))[0]
    base = base.replace("_detailed", "")
    
    # Collect all databases that need ephemeral copies
    databases_needing_ephemeral = set()
    for item in predictions:
        op_type = item.get("metadata", {}).get("original_item", {}).get("operation", "unknown")
        if op_type and op_type.upper() in MODIFYING_OPERATIONS:
            db_id = str(item.get("database", "")).strip()
            if db_id:
                databases_needing_ephemeral.add(db_id)
    
    # Create ephemeral databases for management operations
    ephemeral_databases = {}
    if databases_needing_ephemeral:
        logger.info(f"Creating ephemeral databases for {len(databases_needing_ephemeral)} databases with management operations...")
        for db_name in databases_needing_ephemeral:
            ephemeral_db = create_ephemeral_db(db_name, worker_id=1, config=pg_config)
            if ephemeral_db:
                ephemeral_databases[db_name] = ephemeral_db
                logger.info(f"  ✓ Created {ephemeral_db} for {db_name}")
            else:
                logger.warning(f"  ✗ Failed to create ephemeral database for {db_name}")
    
    # Initialize statistics
    scores = {op: {"count": 0, "correct": 0, "wrong": 0, "error": 0} for op in OPERATION_TYPES}
    scores["all"] = {"count": 0, "correct": 0, "wrong": 0, "error": 0}
    
    # Open incorrect log file
    incorrect_path = os.path.join(output_dir, f"{base}_baseline_incorrect.txt")
    incorrect_log = open(incorrect_path, "w", encoding="utf-8")
    
    # Track detailed results
    detailed_results = []
    
    logger.info(f"Starting baseline evaluation of {len(predictions)} samples...")
    
    for idx, item in enumerate(predictions):
        # Extract fields
        sample_id = item.get("id", f"sample_{idx}")
        db_id = str(item.get("database", "")).strip()
        gold_sql = str(item.get("gold_sql", "")).strip()
        # Try cleaned_sql first (from prediction), fallback to pred_sql
        pred_sql = str(item.get("cleaned_sql", item.get("pred_sql", ""))).strip()
        
        # Get operation type from dataset metadata (same as crud_level.py)
        op_type = item.get("metadata", {}).get("original_item", {}).get("operation", "unknown")
        # Normalize to uppercase
        if op_type and op_type != "unknown":
            op_type = op_type.upper()
            # Map Analyze to standard name
            if op_type == "ANALYZE":
                op_type = "Analyze"
        
        # Update counts
        scores[op_type]["count"] += 1
        scores["all"]["count"] += 1
        
        result_entry = {
            "index": idx + 1,
            "id": sample_id,
            "db_id": db_id,
            "operation": op_type,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
        }
        
        # Skip if gold SQL is invalid
        if not gold_sql or "sorry" in gold_sql.lower():
            logger.warning(f"Sample {idx}: Invalid gold SQL, skipping")
            scores[op_type]["count"] -= 1
            scores["all"]["count"] -= 1
            continue
        
        # Check if prediction is empty or refusal
        if not pred_sql or ("sorry" in pred_sql.lower() and "cannot" in pred_sql.lower()):
            scores[op_type]["wrong"] += 1
            scores["all"]["wrong"] += 1
            result_entry["result"] = "empty_or_refusal"
            result_entry["correct"] = False
            detailed_results.append(result_entry)
            incorrect_log.write(
                f"index: {idx + 1}\n"
                f"db_id: {db_id}\n"
                f"operation: {op_type}\n"
                f"pred: {pred_sql[:200] if pred_sql else '<empty>'}\n"
                f"gold: {gold_sql[:200]}\n\n"
            )
            continue
        
        # Determine if this is a modifying operation
        is_modifying = op_type in MODIFYING_OPERATIONS
        
        # Select database: ephemeral for management ops, original for queries
        if is_modifying and db_id in ephemeral_databases:
            execution_db = ephemeral_databases[db_id]
            result_entry["used_ephemeral_db"] = True
        else:
            execution_db = db_id
            result_entry["used_ephemeral_db"] = False
        
        # Preprocess SQLs: remove ROUND functions to avoid PostgreSQL type errors
        gold_sql_clean = remove_round_functions(gold_sql)
        pred_sql_clean = remove_round_functions(pred_sql)
        
        # Execute both SQLs and compare
        gold_success, gold_result, gold_error = execute_sql_postgres(
            gold_sql_clean, execution_db, pg_config, is_modifying=is_modifying
        )
        pred_success, pred_result, pred_error = execute_sql_postgres(
            pred_sql_clean, execution_db, pg_config, is_modifying=is_modifying
        )
        
        # Reset ephemeral database after each management operation
        if is_modifying and db_id in ephemeral_databases:
            reset_ephemeral_database(ephemeral_databases[db_id], pg_config)
        
        result_entry["gold_success"] = gold_success
        result_entry["pred_success"] = pred_success
        
        if not gold_success:
            # Gold SQL failed - mark as error (shouldn't happen)
            scores[op_type]["error"] += 1
            scores["all"]["error"] += 1
            result_entry["result"] = "gold_error"
            result_entry["correct"] = False
            result_entry["error"] = gold_error
        elif not pred_success:
            # Prediction SQL failed
            scores[op_type]["wrong"] += 1
            scores["all"]["wrong"] += 1
            result_entry["result"] = "pred_error"
            result_entry["correct"] = False
            result_entry["error"] = pred_error
            incorrect_log.write(
                f"index: {idx + 1}\n"
                f"db_id: {db_id}\n"
                f"operation: {op_type}\n"
                f"result: execution_error\n"
                f"error: {pred_error}\n"
                f"pred: {pred_sql[:200]}\n"
                f"gold: {gold_sql[:200]}\n\n"
            )
        elif compare_results(pred_result, gold_result):
            # Results match
            scores[op_type]["correct"] += 1
            scores["all"]["correct"] += 1
            result_entry["result"] = "correct"
            result_entry["correct"] = True
        else:
            # Results don't match
            scores[op_type]["wrong"] += 1
            scores["all"]["wrong"] += 1
            result_entry["result"] = "wrong"
            result_entry["correct"] = False
            incorrect_log.write(
                f"index: {idx + 1}\n"
                f"db_id: {db_id}\n"
                f"operation: {op_type}\n"
                f"result: wrong_result\n"
                f"pred: {pred_sql[:200]}\n"
                f"gold: {gold_sql[:200]}\n\n"
            )
        
        detailed_results.append(result_entry)
        
        # Progress logging
        if (idx + 1) % 100 == 0:
            current_acc = scores["all"]["correct"] / scores["all"]["count"] * 100 if scores["all"]["count"] > 0 else 0
            logger.info(f"Evaluated {idx + 1}/{len(predictions)} samples, current EX: {current_acc:.2f}%")
    
    incorrect_log.close()
    
    # Cleanup ephemeral databases
    if ephemeral_databases:
        logger.info("Cleaning up ephemeral databases...")
        cleanup_ephemeral_databases(pg_config)
    
    # Calculate and print results
    print_and_save_results(scores, output_dir, base, detailed_results)
    
    return {
        "scores": scores,
        "detailed_results": detailed_results,
    }


def print_and_save_results(
    scores: Dict[str, Dict],
    output_dir: str,
    base_name: str,
    detailed_results: List[Dict],
):
    """Print and save evaluation results."""
    
    # Calculate overall accuracy
    all_bucket = scores["all"]
    total = all_bucket["count"]
    correct = all_bucket["correct"]
    accuracy = correct / total * 100 if total > 0 else 0
    
    # Build result text
    lines = []
    lines.append("=" * 70)
    lines.append("LIVESQLBENCH-FULL BASELINE TEXT2SQL EVALUATION")
    lines.append("=" * 70)
    lines.append(f"Prediction file: {base_name}")
    lines.append(f"Total Samples: {total}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("EXECUTION ACCURACY (EX) BY OPERATION")
    lines.append("-" * 70)
    lines.append(f"{'Operation':<12} {'Count':>8} {'Correct':>8} {'Wrong':>8} {'Error':>8} {'Accuracy':>10}")
    lines.append("-" * 70)
    
    for op in OPERATION_TYPES:
        bucket = scores[op]
        count = bucket["count"]
        if count > 0:
            acc = bucket["correct"] / count * 100
            lines.append(
                f"{op:<12} {count:>8} {bucket['correct']:>8} {bucket['wrong']:>8} {bucket['error']:>8} {acc:>9.2f}%"
            )
    
    lines.append("-" * 70)
    lines.append(
        f"{'all':<12} {total:>8} {correct:>8} {all_bucket['wrong']:>8} {all_bucket['error']:>8} {accuracy:>9.2f}%"
    )
    lines.append("=" * 70)
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Total samples: {total}")
    lines.append(f"  Correct: {correct}")
    lines.append(f"  Wrong: {all_bucket['wrong']}")
    lines.append(f"  Error: {all_bucket['error']}")
    lines.append(f"  Overall Execution Accuracy (EX): {accuracy:.2f}%")
    lines.append("=" * 70)
    
    result_text = "\n".join(lines)
    
    # Print to console
    print(result_text)
    
    # Save to text file
    txt_path = os.path.join(output_dir, f"{base_name}_baseline_evaluate_result.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    logger.info(f"Results saved to: {txt_path}")
    
    # Save to JSON file
    json_path = os.path.join(output_dir, f"{base_name}_baseline_evaluate_result.json")
    json_result = {
        "scores": scores,
        "overall_accuracy": accuracy,
        "total": total,
        "correct": correct,
        "timestamp": datetime.now().isoformat(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=2)
    logger.info(f"JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LiveSQLBench-Full Baseline Text2SQL Evaluation (PostgreSQL)"
    )
    
    parser.add_argument("--pred", required=True,
                        help="Path to prediction file (.sql or _detailed.json)")
    
    # PostgreSQL configuration
    parser.add_argument("--db_host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--db_port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--db_user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--db_password", required=True, help="PostgreSQL password")
    
    # Database reinitialization
    parser.add_argument("--reinit_db", action="store_true",
                        help="Reinitialize databases from dumps before evaluation (prevents pollution)")
    parser.add_argument("--dumps_dir", default="data/livesqlbench-full-postgresql/bird-interact-full-dumps",
                        help="Directory containing database dumps for reinitialization")
    
    # Output options
    parser.add_argument("--output_dir", default=None, help="Output directory")
    
    args = parser.parse_args()
    
    pg_config = {
        "host": args.db_host,
        "port": args.db_port,
        "user": args.db_user,
        "password": args.db_password,
    }
    
    # Reinitialize databases if requested
    if args.reinit_db:
        logger.info("Reinitializing databases before evaluation...")
        success, failed = reinitialize_databases(pg_config, args.dumps_dir)
        if failed > 0:
            logger.warning(f"{failed} databases failed to reinitialize")
        else:
            logger.info("All databases reinitialized successfully")
    
    evaluate_baseline_livesqlbench(
        prediction_path=args.pred,
        pg_config=pg_config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
