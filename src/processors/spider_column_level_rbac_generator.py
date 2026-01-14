"""
Column-Level RBAC Dataset Generator for Spider Benchmark.

This is a self-contained module for generating column-level RBAC datasets for Spider.
It includes all necessary functionality for SQL parsing, difficulty computation, and
dataset generation.

Key features:
1. Parse Spider schema from schema.sql files (with SQLite fallback)
2. Compute SQL difficulty from query complexity analysis
3. Generate column-level RBAC datasets with permission checking
"""

import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime

from nltk import word_tokenize

from .column_extractor import SQLColumnExtractor, extract_query_columns
from .column_permission_checker import ColumnPermissionChecker, check_column_permission


# ===========================================================================
# Constants
# ===========================================================================

# Denial message template - consistent with Bird
DENIAL_MESSAGE = "Sorry, I cannot answer."

# Spider difficulty levels
DIFFICULTY_LEVELS = ["easy", "medium", "hard", "extra"]

# SQL parsing constants
CLAUSE_KEYWORDS = (
    "select", "from", "where", "group", "order", "limit",
    "intersect", "union", "except",
)
JOIN_KEYWORDS = ("join", "on", "as")
WHERE_OPS = (
    "not", "between", "=", ">", "<", ">=", "<=", "!=", 
    "in", "like", "is", "exists",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
TABLE_TYPE = {"sql": "sql", "table_unit": "table_unit"}
COND_OPS = ("and", "or")
SQL_OPS = ("intersect", "union", "except")
ORDER_OPS = ("desc", "asc")

# Instruction template - adapted for Spider (no evidence field)
INSTRUCTION_TEMPLATE = """##Instruction:
Database: {db_id}
Total Tables: {total_tables}
Total Columns: {total_columns}

Table Schemas:

{schema_text}

##Role Access Policy (Column-Level):
Role: {role}
Accessible Columns: {policy_text}

Based on the above schema and your role's column-level access policy, generate a SQL query to answer the question. If your role does not have access to the required columns, respond with: "Sorry, I cannot answer."
"""


# ===========================================================================
# Schema Class for SQL Parsing
# ===========================================================================

class Schema:
    """Schema mapping table & column names to unique identifiers for SQL parsing."""

    def __init__(self, schema: Dict[str, List[str]]):
        self._schema = schema
        self._id_map = self._create_id_map(schema)

    @property
    def schema(self) -> Dict[str, List[str]]:
        return self._schema

    @property
    def idMap(self) -> Dict[str, str]:
        return self._id_map

    def _create_id_map(self, schema: Dict[str, List[str]]) -> Dict[str, str]:
        id_map: Dict[str, str] = {"*": "__all__"}
        identifier = 1
        for table, columns in schema.items():
            for column in columns:
                id_map[f"{table.lower()}.{column.lower()}"] = (
                    f"__{table.lower()}.{column.lower()}__"
                )
                identifier += 1

        for table in schema:
            id_map[table.lower()] = f"__{table.lower()}__"
            identifier += 1

        return id_map


# ===========================================================================
# Schema Loading Functions
# ===========================================================================

def get_schema_from_db(db_path: str) -> Dict[str, List[str]]:
    """Load table → column mapping from SQLite database."""
    schema: Dict[str, List[str]] = {}
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [str(result[0].lower()) for result in cursor.fetchall()]
        for table in tables:
            cursor.execute(f"PRAGMA table_info(`{table}`)")
            schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]
    finally:
        conn.close()
    return schema


def format_policy_text(policy: Dict[str, List[str]]) -> str:
    """Format policy dict to readable text."""
    parts = []
    for table, columns in policy.items():
        if columns == ["*"] or columns == "*":
            parts.append(f"{table}: ALL COLUMNS")
        else:
            parts.append(f"{table}: {', '.join(columns)}")
    return "; ".join(parts)


# ===========================================================================
# SQL Tokenization and Parsing Functions
# ===========================================================================

def tokenize(sql: str) -> List[str]:
    """Tokenize SQL query for parsing."""
    sql = str(sql).replace("'", '"')
    quote_indices = [idx for idx, char in enumerate(sql) if char == '"']
    if len(quote_indices) % 2 != 0:
        raise ValueError("Unbalanced quotes in SQL query")

    preserved_literals: Dict[str, str] = {}
    for index in range(len(quote_indices) - 1, 0, -2):
        start = quote_indices[index - 1]
        end = quote_indices[index]
        literal = sql[start : end + 1]
        placeholder = f"__val_{start}_{end}__"
        sql = sql[:start] + placeholder + sql[end + 1 :]
        preserved_literals[placeholder] = literal

    tokens = [word.lower() for word in word_tokenize(sql)]
    for position, token in enumerate(tokens):
        if token in preserved_literals:
            tokens[position] = preserved_literals[token]

    equality_indices = [idx for idx, tok in enumerate(tokens) if tok == "="]
    equality_indices.reverse()
    for idx in equality_indices:
        if tokens[idx - 1] in ("!", ">", "<"):
            tokens = tokens[: idx - 1] + [tokens[idx - 1] + "="] + tokens[idx + 1 :]

    return tokens


def scan_alias(tokens: List[str]) -> Dict[str, str]:
    """Scan for table aliases in SQL tokens."""
    alias_map: Dict[str, str] = {}
    as_positions = [idx for idx, tok in enumerate(tokens) if tok == "as"]
    for idx in as_positions:
        alias_map[tokens[idx + 1]] = tokens[idx - 1]
    return alias_map


def get_tables_with_alias(schema: Dict[str, List[str]], tokens: List[str]) -> Dict[str, str]:
    """Get table name to alias mapping."""
    tables = scan_alias(tokens)
    for name in schema:
        if name in tables:
            raise ValueError(f"Alias {name} has the same name as a table")
        tables[name] = name
    return tables


def parse_col(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, str]:
    """Parse a column reference from tokens."""
    token = tokens[start_idx]
    if token == "*":
        return start_idx + 1, schema.idMap[token]

    if "." in token:
        alias, column = token.split(".", 1)
        key = f"{tables_with_alias[alias]}.{column}"
        return start_idx + 1, schema.idMap[key]

    if not default_tables:
        raise AssertionError("Default tables should not be empty when parsing a column")

    for alias in default_tables:
        table = tables_with_alias[alias]
        if token in schema.schema[table]:
            key = f"{table}.{token}"
            return start_idx + 1, schema.idMap[key]

    raise AssertionError(f"Error parsing column token: {token}")


def parse_col_unit(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Tuple[int, str, bool]]:
    """Parse a column unit (possibly with aggregation)."""
    idx = start_idx
    length = len(tokens)
    is_block = False
    is_distinct = False

    if tokens[idx] == "(":
        is_block = True
        idx += 1

    if tokens[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(tokens[idx])
        idx += 1
        assert idx < length and tokens[idx] == "("
        idx += 1
        if tokens[idx] == "distinct":
            idx += 1
            is_distinct = True
        idx, col_id = parse_col(tokens, idx, tables_with_alias, schema, default_tables)
        assert idx < length and tokens[idx] == ")"
        idx += 1
        return idx, (agg_id, col_id, is_distinct)

    if tokens[idx] == "distinct":
        idx += 1
        is_distinct = True

    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(tokens, idx, tables_with_alias, schema, default_tables)

    if is_block:
        assert tokens[idx] == ")"
        idx += 1

    return idx, (agg_id, col_id, is_distinct)


def has_agg(unit: Tuple[int, Any, Any]) -> bool:
    """Check if a column unit has aggregation."""
    return unit[0] != AGG_OPS.index("none")


def parse_val_unit(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Tuple[int, Tuple[int, str, bool], Optional[Tuple[int, str, bool]]]]:
    """Parse a value unit."""
    idx = start_idx
    length = len(tokens)
    is_block = False

    if tokens[idx] == "(":
        is_block = True
        idx += 1

    idx, col_unit1 = parse_col_unit(tokens, idx, tables_with_alias, schema, default_tables)
    unit_op = UNIT_OPS.index("none")
    col_unit2: Optional[Tuple[int, str, bool]] = None

    if idx < length and tokens[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(tokens[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(tokens, idx, tables_with_alias, schema, default_tables)

    if is_block:
        assert tokens[idx] == ")"
        idx += 1

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
) -> Tuple[int, str, str]:
    """Parse a table unit."""
    idx = start_idx
    key = tables_with_alias[tokens[idx]]

    if idx + 1 < len(tokens) and tokens[idx + 1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Any]:
    """Parse a value (literal, column, or subquery)."""
    idx = start_idx
    length = len(tokens)
    is_block = False

    if tokens[idx] == "(":
        is_block = True
        idx += 1

    if tokens[idx] == "select":
        idx, val = parse_sql(tokens, idx, tables_with_alias, schema)
    elif '"' in tokens[idx]:
        val = tokens[idx]
        idx += 1
    else:
        try:
            val = float(tokens[idx])
            idx += 1
        except ValueError:
            end_idx = idx
            while (
                end_idx < length
                and tokens[end_idx] not in {",", ")", "and"}
                and tokens[end_idx] not in CLAUSE_KEYWORDS
                and tokens[end_idx] not in JOIN_KEYWORDS
            ):
                end_idx += 1

            idx, val = parse_col_unit(
                tokens[start_idx:end_idx],
                0,
                tables_with_alias,
                schema,
                default_tables,
            )
            idx = end_idx

    if is_block:
        assert tokens[idx] == ")"
        idx += 1

    return idx, val


def parse_condition(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, List[Any]]:
    """Parse a WHERE/HAVING condition."""
    idx = start_idx
    length = len(tokens)
    conditions: List[Any] = []

    while idx < length:
        idx, val_unit = parse_val_unit(tokens, idx, tables_with_alias, schema, default_tables)
        not_op = False
        if tokens[idx] == "not":
            not_op = True
            idx += 1

        assert idx < length and tokens[idx] in WHERE_OPS, (
            f"Error condition: idx={idx}, token={tokens[idx]}"
        )
        op_id = WHERE_OPS.index(tokens[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index("between"):
            idx, val1 = parse_value(tokens, idx, tables_with_alias, schema, default_tables)
            assert tokens[idx] == "and"
            idx += 1
            idx, val2 = parse_value(tokens, idx, tables_with_alias, schema, default_tables)
        else:
            idx, val1 = parse_value(tokens, idx, tables_with_alias, schema, default_tables)
            val2 = None

        conditions.append((not_op, op_id, val_unit, val1, val2))

        if idx < length and (
            tokens[idx] in CLAUSE_KEYWORDS
            or tokens[idx] in (")", ";")
            or tokens[idx] in JOIN_KEYWORDS
        ):
            break

        if idx < length and tokens[idx] in COND_OPS:
            conditions.append(tokens[idx])
            idx += 1

    return idx, conditions


def parse_select(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Tuple[bool, List[Any]]]:
    """Parse SELECT clause."""
    idx = start_idx
    length = len(tokens)

    assert tokens[idx] == "select"
    idx += 1
    is_distinct = False
    if idx < length and tokens[idx] == "distinct":
        idx += 1
        is_distinct = True

    val_units: List[Any] = []

    while idx < length and tokens[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if tokens[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(tokens[idx])
            idx += 1
        idx, val_unit = parse_val_unit(tokens, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < length and tokens[idx] == ",":
            idx += 1

    return idx, (is_distinct, val_units)


def parse_from(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
) -> Tuple[int, List[Tuple[str, Any]], List[Any], List[str]]:
    """Parse FROM clause."""
    assert "from" in tokens[start_idx:]

    length = len(tokens)
    idx = tokens.index("from", start_idx) + 1
    default_tables: List[str] = []
    table_units: List[Tuple[str, Any]] = []
    conditions: List[Any] = []

    while idx < length:
        is_block = False
        if tokens[idx] == "(":
            is_block = True
            idx += 1

        if tokens[idx] == "select":
            idx, sql = parse_sql(tokens, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE["sql"], sql))
        else:
            if idx < length and tokens[idx] == "join":
                idx += 1
            idx, table_unit, table_name = parse_table_unit(tokens, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE["table_unit"], table_unit))
            default_tables.append(table_name)
        if idx < length and tokens[idx] == "on":
            idx += 1
            idx, this_conds = parse_condition(tokens, idx, tables_with_alias, schema, default_tables)
            if conditions:
                conditions.append("and")
            conditions.extend(this_conds)

        if is_block:
            assert tokens[idx] == ")"
            idx += 1
        if idx < length and (tokens[idx] in CLAUSE_KEYWORDS or tokens[idx] in (")", ";")):
            break

    return idx, table_units, conditions, default_tables


def parse_where(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: List[str],
) -> Tuple[int, List[Any]]:
    """Parse WHERE clause."""
    idx = start_idx
    length = len(tokens)

    if idx >= length or tokens[idx] != "where":
        return idx, []

    idx += 1
    idx, conditions = parse_condition(tokens, idx, tables_with_alias, schema, default_tables)
    return idx, conditions


def parse_group_by(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: List[str],
) -> Tuple[int, List[Tuple[int, str]]]:
    """Parse GROUP BY clause."""
    idx = start_idx
    length = len(tokens)
    columns: List[Tuple[int, str]] = []

    if idx >= length or tokens[idx] != "group":
        return idx, columns

    idx += 1
    assert tokens[idx] == "by"
    idx += 1

    while idx < length and tokens[idx] not in CLAUSE_KEYWORDS and tokens[idx] not in (")", ";"):
        idx, col_unit = parse_col_unit(tokens, idx, tables_with_alias, schema, default_tables)
        columns.append(col_unit)
        if idx < length and tokens[idx] == ",":
            idx += 1
        else:
            break

    return idx, columns


def parse_order_by(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: List[str],
) -> Tuple[int, Tuple[str, List[Any]]]:
    """Parse ORDER BY clause."""
    idx = start_idx
    length = len(tokens)
    val_units: List[Any] = []
    order_type = "asc"

    if idx >= length or tokens[idx] != "order":
        return idx, val_units

    idx += 1
    assert tokens[idx] == "by"
    idx += 1

    while idx < length and tokens[idx] not in CLAUSE_KEYWORDS and tokens[idx] not in (")", ";"):
        idx, val_unit = parse_val_unit(tokens, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < length and tokens[idx] in ORDER_OPS:
            order_type = tokens[idx]
            idx += 1
        if idx < length and tokens[idx] == ",":
            idx += 1
        else:
            break

    return idx, (order_type, val_units)


def parse_having(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: List[str],
) -> Tuple[int, List[Any]]:
    """Parse HAVING clause."""
    idx = start_idx
    length = len(tokens)

    if idx >= length or tokens[idx] != "having":
        return idx, []

    idx += 1
    idx, conditions = parse_condition(tokens, idx, tables_with_alias, schema, default_tables)
    return idx, conditions


def parse_limit(tokens: List[str], start_idx: int) -> Tuple[int, Optional[int]]:
    """Parse LIMIT clause."""
    idx = start_idx
    length = len(tokens)

    if idx < length and tokens[idx] == "limit":
        idx += 2
        if not isinstance(tokens[idx - 1], int):
            return idx, 1
        return idx, int(tokens[idx - 1])

    return idx, None


def skip_semicolon(tokens: List[str], start_idx: int) -> int:
    """Skip semicolons in token list."""
    idx = start_idx
    while idx < len(tokens) and tokens[idx] == ";":
        idx += 1
    return idx


def parse_sql(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
) -> Tuple[int, Dict[str, Any]]:
    """Parse complete SQL statement."""
    is_block = False
    length = len(tokens)
    idx = start_idx

    sql: Dict[str, Any] = {}
    if tokens[idx] == "(":
        is_block = True
        idx += 1

    from_end_idx, table_units, conditions, default_tables = parse_from(
        tokens, start_idx, tables_with_alias, schema
    )
    sql["from"] = {"table_units": table_units, "conds": conditions}

    _, select_units = parse_select(tokens, idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql["select"] = select_units

    idx, where_conds = parse_where(tokens, idx, tables_with_alias, schema, default_tables)
    sql["where"] = where_conds

    idx, group_units = parse_group_by(tokens, idx, tables_with_alias, schema, default_tables)
    sql["groupBy"] = group_units

    idx, having_conds = parse_having(tokens, idx, tables_with_alias, schema, default_tables)
    sql["having"] = having_conds

    idx, order_units = parse_order_by(tokens, idx, tables_with_alias, schema, default_tables)
    sql["orderBy"] = order_units

    idx, limit_val = parse_limit(tokens, idx)
    sql["limit"] = limit_val

    idx = skip_semicolon(tokens, idx)
    if is_block:
        assert tokens[idx] == ")"
        idx += 1
    idx = skip_semicolon(tokens, idx)

    for op in SQL_OPS:
        sql[op] = None
    if idx < length and tokens[idx] in SQL_OPS:
        sql_op = tokens[idx]
        idx += 1
        idx, nested_sql = parse_sql(tokens, idx, tables_with_alias, schema)
        sql[sql_op] = nested_sql

    return idx, sql


def get_sql(schema: Schema, query: str) -> Dict[str, Any]:
    """Parse SQL query string into structured dictionary."""
    tokens = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, tokens)
    _, sql = parse_sql(tokens, 0, tables_with_alias, schema)
    return sql


# ===========================================================================
# Difficulty Computation Functions
# ===========================================================================

def get_nestedSQL(sql: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get nested SQL queries from a parsed SQL dictionary."""
    nested: List[Dict[str, Any]] = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
        if isinstance(cond_unit[4], dict):
            nested.append(cond_unit[4])
    if sql["intersect"] is not None:
        nested.append(sql["intersect"])
    if sql["except"] is not None:
        nested.append(sql["except"])
    if sql["union"] is not None:
        nested.append(sql["union"])
    return nested


def count_agg(units: List[Any]) -> int:
    """Count aggregation functions in column units."""
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql: Dict[str, Any]) -> int:
    """Count component 1 complexity factors."""
    count = 0
    if sql["where"]:
        count += 1
    if sql["groupBy"]:
        count += 1
    if sql["orderBy"]:
        count += 1
    if sql["limit"] is not None:
        count += 1
    if sql["from"]["table_units"]:
        count += len(sql["from"]["table_units"]) - 1

    ao_tokens = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += len([token for token in ao_tokens if token == "or"])

    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += len([cond for cond in cond_units if cond[1] == WHERE_OPS.index("like")])

    return count


def count_component2(sql: Dict[str, Any]) -> int:
    """Count component 2 complexity factors (nested queries)."""
    return len(get_nestedSQL(sql))


def count_others(sql: Dict[str, Any]) -> int:
    """Count other complexity factors."""
    count = 0
    agg_count = count_agg(sql["select"][1])
    agg_count += count_agg(sql["where"][::2])
    agg_count += count_agg(sql["groupBy"])
    if sql["orderBy"]:
        agg_count += count_agg(
            [unit[1] for unit in sql["orderBy"][1] if unit[1]]
            + [unit[2] for unit in sql["orderBy"][1] if unit[2]]
        )
    agg_count += count_agg(sql["having"])
    if agg_count > 1:
        count += 1

    if len(sql["select"][1]) > 1:
        count += 1
    if len(sql["where"]) > 1:
        count += 1
    if len(sql["groupBy"]) > 1:
        count += 1

    return count


def compute_spider_difficulty(sql_dict: Dict[str, Any]) -> str:
    """
    Compute Spider difficulty category from parsed SQL dictionary.
    
    Categories:
    - easy: Simple queries with minimal complexity
    - medium: Queries with moderate complexity
    - hard: Complex queries with multiple features
    - extra: Very complex queries with nested queries or many features
    """
    comp1 = count_component1(sql_dict)
    comp2 = count_component2(sql_dict)
    others = count_others(sql_dict)

    if comp1 <= 1 and others == 0 and comp2 == 0:
        return "easy"
    if ((others <= 2 and comp1 <= 1 and comp2 == 0) or (comp1 <= 2 and others < 2 and comp2 == 0)):
        return "medium"
    if (
        (others > 2 and comp1 <= 2 and comp2 == 0)
        or (2 < comp1 <= 3 and others <= 2 and comp2 == 0)
        or (comp1 <= 1 and others == 0 and comp2 <= 1)
    ):
        return "hard"
    return "extra"


# ===========================================================================
# Schema Parsing from schema.sql
# ===========================================================================

TABLE_PATTERN = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?([A-Za-z0-9_]+)[`\"\]]?",
    re.IGNORECASE
)

COLUMN_PATTERN = re.compile(
    r"^\s*[`\"\[]?([A-Za-z0-9_]+)[`\"\]]?\s+(?:INTEGER|TEXT|REAL|BLOB|VARCHAR|INT|CHAR|DECIMAL|DATE|DATETIME|BOOLEAN|FLOAT|DOUBLE|NUMERIC|BIGINT|SMALLINT|TINYINT)",
    re.IGNORECASE | re.MULTILINE
)


def parse_schema_sql(schema_path: Path) -> Dict[str, List[str]]:
    """
    Parse CREATE TABLE statements from schema.sql file.
    
    Returns:
        Dict mapping table names to column lists
    """
    schema: Dict[str, List[str]] = {}
    
    try:
        schema_text = schema_path.read_text(encoding='utf-8')
    except Exception:
        return schema
    
    # Split by CREATE TABLE statements
    # Find all CREATE TABLE blocks
    table_matches = list(TABLE_PATTERN.finditer(schema_text))
    
    for i, match in enumerate(table_matches):
        table_name = match.group(1).strip('`"[]').lower()
        
        # Find the block between this CREATE TABLE and the next (or end)
        start_pos = match.end()
        if i + 1 < len(table_matches):
            end_pos = table_matches[i + 1].start()
        else:
            end_pos = len(schema_text)
        
        block = schema_text[start_pos:end_pos]
        
        # Find opening and closing parentheses
        paren_start = block.find('(')
        if paren_start == -1:
            continue
            
        # Find matching closing paren
        depth = 0
        paren_end = -1
        for j, char in enumerate(block[paren_start:], paren_start):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    paren_end = j
                    break
        
        if paren_end == -1:
            continue
        
        columns_block = block[paren_start+1:paren_end]
        
        # Parse column definitions
        columns = []
        seen_columns = set()  # Track unique columns
        for line in columns_block.split(','):
            line = line.strip()
            if not line:
                continue
            
            # Skip constraints
            upper_line = line.upper()
            if any(kw in upper_line for kw in ['PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'CONSTRAINT']):
                continue
            
            # Extract column name (first word)
            col_match = re.match(r'^[`\"\[]?([A-Za-z0-9_]+)[`\"\]]?', line)
            if col_match:
                col_name = col_match.group(1).lower()
                # Validate it's not a keyword and not duplicate
                if col_name.upper() not in ['PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'CONSTRAINT', 'INDEX', 'KEY']:
                    if col_name not in seen_columns:
                        columns.append(col_name)
                        seen_columns.add(col_name)
        
        if columns:
            schema[table_name] = columns
    
    return schema


# ---------------------------------------------------------------------------
# Dataset Entry Structure
# ---------------------------------------------------------------------------

@dataclass 
class RBACDatasetEntry:
    """A single entry in the RBAC dataset (matches Bird format)."""
    db_id: str
    instruction: str  # Full instruction with schema and policy
    role: str
    policy: Dict[str, List[str]]  # Column-level policy: {table: [columns]}
    input: str  # User question
    output: str  # Either SQL or denial message
    difficulty: str
    metadata: Dict[str, Any]  # Gold SQL, permission status, etc.
    
    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main Generator Class
# ---------------------------------------------------------------------------

class SpiderColumnLevelRBACGenerator:
    """Generate column-level RBAC datasets for Spider benchmark."""
    
    def __init__(
        self,
        role_assignments_path: str,
        spider_data_path: str,
        spider_db_root: str,
        project_root: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize generator.
        
        Args:
            role_assignments_path: Path to column_level_role_assignments_spider.json
            spider_data_path: Path to Spider dev.json or test.json
            spider_db_root: Root directory containing Spider databases
            project_root: Project root (optional)
            seed: Random seed for reproducibility
        """
        self.role_assignments_path = Path(role_assignments_path)
        self.spider_data_path = Path(spider_data_path)
        self.spider_db_root = Path(spider_db_root)
        self.project_root = Path(project_root) if project_root else self.spider_db_root.parent.parent
        self.seed = seed
        
        random.seed(seed)
        
        # Load data
        self.role_assignments = self._load_role_assignments()
        self.spider_data = self._load_spider_data()
        
        # Index queries by db_id
        self.queries_by_db = self._index_queries_by_db()
        
        # Load tables.json for schema metadata (PK/FK)
        self.tables_info = self._load_tables_info()
        
        # Initialize components
        self.checker = ColumnPermissionChecker()
        
        # Cache for database schemas and difficulties
        self._schema_cache: Dict[str, Dict[str, List[str]]] = {}
        self._schema_text_cache: Dict[str, Tuple[str, int, int]] = {}
        self._difficulty_cache: Dict[Tuple[str, str], str] = {}
    
    def _load_tables_info(self) -> Dict[str, Dict]:
        """Load tables.json for schema metadata (primary keys, foreign keys)."""
        # Try to find tables.json in the spider data directory
        spider_root = self.spider_db_root.parent
        tables_file = spider_root / "tables.json"
        
        if not tables_file.exists():
            # Fallback to test_tables.json if using test data
            tables_file = spider_root / "test_tables.json"
        
        if not tables_file.exists():
            return {}
        
        try:
            with open(tables_file, 'r', encoding='utf-8') as f:
                tables_list = json.load(f)
            # Convert to dict indexed by db_id
            return {item['db_id']: item for item in tables_list}
        except Exception:
            return {}
    
    def _load_role_assignments(self) -> Dict:
        """Load role assignments from JSON."""
        with open(self.role_assignments_path, 'r') as f:
            data = json.load(f)
        return data.get("assignments", data)
    
    def _load_spider_data(self) -> List[Dict]:
        """Load Spider dataset (dev.json or test.json)."""
        with open(self.spider_data_path, 'r') as f:
            return json.load(f)
    
    def _index_queries_by_db(self) -> Dict[str, List[Dict]]:
        """Index queries by database ID."""
        index = {}
        for item in self.spider_data:
            db_id = item.get('db_id', '')
            if db_id not in index:
                index[db_id] = []
            index[db_id].append(item)
        return index
    
    def get_db_path(self, db_id: str) -> str:
        """Get full path to database file."""
        return str(self.spider_db_root / db_id / f"{db_id}.sqlite")
    
    def get_schema_path(self, db_id: str) -> Path:
        """Get path to schema.sql file."""
        return self.spider_db_root / db_id / "schema.sql"
    
    def validate_database(self, db_id: str) -> Tuple[bool, str]:
        """
        Validate that a database has required files.
        
        Schema can be obtained from either schema.sql OR SQLite file directly.
        
        Returns:
            (is_valid, reason)
        """
        db_dir = self.spider_db_root / db_id
        if not db_dir.exists():
            return False, "Database directory not found"
        
        sqlite_files = list(db_dir.glob("*.sqlite"))
        if not sqlite_files:
            return False, "No .sqlite file found"
        
        # Schema can come from schema.sql OR be extracted from SQLite
        schema_path = db_dir / "schema.sql"
        if not schema_path.exists():
            # Verify we can extract schema from SQLite
            try:
                db_path = str(sqlite_files[0])
                schema = get_schema_from_db(db_path)
                if not schema:
                    return False, "Cannot extract schema from SQLite"
            except Exception as e:
                return False, f"SQLite schema extraction failed: {e}"
        
        return True, "OK"
    
    def get_valid_databases(self) -> List[str]:
        """Get list of databases with valid schema and sqlite files."""
        valid_dbs = []
        for db_dir in sorted(self.spider_db_root.iterdir()):
            if not db_dir.is_dir() or db_dir.name.startswith('.'):
                continue
            is_valid, _ = self.validate_database(db_dir.name)
            if is_valid:
                valid_dbs.append(db_dir.name)
        return valid_dbs
    
    def get_schema(self, db_id: str) -> Dict[str, List[str]]:
        """Get schema for a database, with caching.
        
        Prioritizes SQLite database as the authoritative source to avoid
        schema.sql parsing issues (e.g., missing columns with inline PRIMARY KEY).
        """
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]
        
        # Prioritize SQLite database as the authoritative source
        # schema.sql parsing has known issues (e.g., skipping columns with inline PRIMARY KEY)
        db_path = self.get_db_path(db_id)
        schema = get_schema_from_db(db_path)
        
        # Fallback to schema.sql only if SQLite fails
        if not schema:
            schema_path = self.get_schema_path(db_id)
            schema = parse_schema_sql(schema_path)
        
        self._schema_cache[db_id] = schema
        return schema
    
    def get_schema_text(self, db_id: str) -> Tuple[str, int, int]:
        """
        Get formatted schema text for a database.
        
        Generates instruction text similar to spider_role_sql_generator.generate_database_instruction(),
        including table names, columns, primary keys, and foreign keys.
        """
        if db_id in self._schema_text_cache:
            return self._schema_text_cache[db_id]
        
        schema = self.get_schema(db_id)
        table_info = self.tables_info.get(db_id)
        
        total_tables = len(schema)
        total_columns = sum(len(cols) for cols in schema.values())
        
        # If we have tables.json metadata, use detailed format like spider_role_sql_generator
        if table_info:
            schema_text = self._generate_detailed_schema_text(db_id, table_info)
        else:
            # Fallback to simple schema list
            schema_text = self._generate_simple_schema_text(db_id, schema)
        
        self._schema_text_cache[db_id] = (schema_text, total_tables, total_columns)
        return schema_text, total_tables, total_columns
    
    def _generate_detailed_schema_text(self, db_id: str, table_info: Dict) -> str:
        """
        Generate detailed schema text with PK/FK info.
        
        Follows the same format as spider_role_sql_generator.generate_database_instruction().
        """
        try:
            tables = table_info["table_names_original"]
            columns = table_info["column_names_original"][1:]  # Skip first [-1, "*"]
            primary_keys = table_info.get("primary_keys", [])
            foreign_keys = table_info.get("foreign_keys", [])
            
            # Build instruction text
            lines = [f"{db_id} contains tables such as " + ", ".join(tables) + "."]
            
            # Add column info for each table
            for i, table_name in enumerate(tables):
                table_columns = [col[1] for col in columns if col[0] == i]
                if table_columns:
                    lines.append(f"Table {table_name} has columns such as " + ", ".join(table_columns) + ".")
                    
                    # Add primary key info
                    for pk in primary_keys:
                        if isinstance(pk, int):
                            if pk > 0 and pk <= len(columns) and columns[pk - 1][0] == i:
                                lines.append(f"{columns[pk - 1][1]} is the primary key.")
                        elif isinstance(pk, list):
                            keys = [columns[k - 1][1] for k in pk 
                                   if k > 0 and k <= len(columns) and columns[k - 1][0] == i]
                            if keys:
                                lines.append(f"The combination of ({', '.join(keys)}) are the primary key.")
            
            # Add foreign key info
            for fk in foreign_keys:
                try:
                    if len(fk) >= 2:
                        fk_col_idx = fk[0] - 1
                        ref_col_idx = fk[1] - 1
                        if 0 <= fk_col_idx < len(columns) and 0 <= ref_col_idx < len(columns):
                            fk_col = columns[fk_col_idx][1]
                            fk_table = tables[columns[fk_col_idx][0]]
                            ref_col = columns[ref_col_idx][1]
                            ref_table = tables[columns[ref_col_idx][0]]
                            lines.append(f"The {fk_col} of {fk_table} is the foreign key of {ref_col} of {ref_table}.")
                except (IndexError, KeyError):
                    continue
            
            return "\n".join(lines)
            
        except Exception:
            # Fallback to simple format
            schema = self.get_schema(db_id)
            return self._generate_simple_schema_text(db_id, schema)
    
    def _generate_simple_schema_text(self, db_id: str, schema: Dict[str, List[str]]) -> str:
        """Generate simple schema text (fallback when tables.json not available)."""
        lines = [f"Database: {db_id}"]
        for table, columns in sorted(schema.items()):
            lines.append(f"Table {table} has columns: " + ", ".join(columns) + ".")
        return "\n".join(lines)
    
    def compute_difficulty(self, db_id: str, sql: str) -> str:
        """
        Compute difficulty for a SQL query.
        
        Uses SQL parsing to determine complexity level (same as spider_role_sql_generator).
        Falls back to 'unknown' if parsing fails.
        """
        cache_key = (db_id, sql)
        if cache_key in self._difficulty_cache:
            return self._difficulty_cache[cache_key]
        
        try:
            schema = self.get_schema(db_id)
            schema_obj = Schema(schema)
            
            # Use get_sql to parse and then compute difficulty
            parsed_sql = get_sql(schema_obj, sql)
            difficulty = compute_spider_difficulty(parsed_sql)
        except Exception:
            difficulty = "unknown"
        
        self._difficulty_cache[cache_key] = difficulty
        return difficulty
    
    def generate_dataset(
        self,
        roles_per_db: Optional[int] = None,
        queries_per_role: Optional[int] = None,
        balance_ratio: float = 0.5
    ) -> List[RBACDatasetEntry]:
        """
        Generate RBAC dataset.
        
        Args:
            roles_per_db: Max roles to use per database (None = use all)
            queries_per_role: Queries per role (None = all queries)
            balance_ratio: Target ratio of allowed vs denied
        
        Returns:
            List of dataset entries
        """
        entries = []
        
        for db_id, roles in self.role_assignments.items():
            if db_id not in self.queries_by_db:
                print(f"Warning: No queries found for database {db_id}")
                continue
            
            # Validate database
            is_valid, reason = self.validate_database(db_id)
            if not is_valid:
                print(f"Warning: Skipping {db_id}: {reason}")
                continue
            
            queries = self.queries_by_db[db_id]
            db_path = self.get_db_path(db_id)
            
            # Select roles
            selected_roles = roles[:roles_per_db] if roles_per_db else roles
            
            for role_info in selected_roles:
                role_entries = self._generate_entries_for_role(
                    db_id=db_id,
                    role_info=role_info,
                    queries=queries,
                    db_path=db_path,
                    max_queries=queries_per_role
                )
                entries.extend(role_entries)
        
        return entries
    
    def _generate_entries_for_role(
        self,
        db_id: str,
        role_info: Dict,
        queries: List[Dict],
        db_path: str,
        max_queries: Optional[int] = None
    ) -> List[RBACDatasetEntry]:
        """Generate entries for a single role."""
        entries = []
        
        role = role_info.get('role', 'Unknown')
        policy = role_info.get('policy', {})
        
        # Get schema text
        schema_text, total_tables, total_columns = self.get_schema_text(db_id)
        policy_text = format_policy_text(policy)
        
        # Select queries
        if max_queries and max_queries < len(queries):
            selected_queries = random.sample(queries, max_queries)
        else:
            selected_queries = queries
        
        for query_item in selected_queries:
            question = query_item.get('question', '')
            gold_sql = query_item.get('query', '')
            
            if not gold_sql:
                continue
            
            # Compute difficulty (Spider doesn't have difficulty in data)
            difficulty = self.compute_difficulty(db_id, gold_sql)
            
            # Check permission
            result = check_column_permission(gold_sql, policy, db_path)
            
            # Determine output
            if result.allowed:
                output = gold_sql
                permission = "allowed"
            else:
                output = DENIAL_MESSAGE
                permission = "denied"
            
            # Build instruction
            instruction = INSTRUCTION_TEMPLATE.format(
                db_id=db_id,
                total_tables=total_tables,
                total_columns=total_columns,
                schema_text=schema_text,
                role=role,
                policy_text=policy_text
            )
            
            # Build metadata
            metadata = {
                "gold_sql": gold_sql,
                "permission": permission,
                "query_columns": result.to_dict()["required_columns"],
                "missing_columns": result.to_dict()["missing_columns"],
                "reason": result.reason,
            }
            
            entry = RBACDatasetEntry(
                db_id=db_id,
                instruction=instruction,
                role=role,
                policy=policy,
                input=question,
                output=output,
                difficulty=difficulty,
                metadata=metadata
            )
            
            entries.append(entry)
        
        return entries
    
    def generate_and_save(
        self,
        output_path: str,
        roles_per_db: Optional[int] = None,
        queries_per_role: Optional[int] = None,
        balance_ratio: float = 0.5
    ) -> Dict:
        """
        Generate dataset and save to file.
        
        Args:
            output_path: Path to save output JSON
            roles_per_db: Number of roles per database (None = all)
            queries_per_role: Queries per role (None = all)
            balance_ratio: Target balance ratio
        
        Returns:
            Summary statistics
        """
        print(f"Generating Spider column-level RBAC dataset...")
        print(f"  Roles per DB: {roles_per_db or 'all'}")
        print(f"  Queries per role: {queries_per_role or 'all'}")
        
        entries = self.generate_dataset(
            roles_per_db=roles_per_db,
            queries_per_role=queries_per_role,
            balance_ratio=balance_ratio
        )
        
        # Calculate statistics
        allowed_count = sum(1 for e in entries if e.metadata.get("permission") == "allowed")
        denied_count = sum(1 for e in entries if e.metadata.get("permission") == "denied")
        
        # Group by database
        by_db = {}
        for e in entries:
            if e.db_id not in by_db:
                by_db[e.db_id] = {"allowed": 0, "denied": 0}
            perm = e.metadata.get("permission", "unknown")
            if perm in by_db[e.db_id]:
                by_db[e.db_id][perm] += 1
        
        # Build output
        output_data = [e.to_dict() for e in entries]
        
        # Save main data file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Save metadata separately
        meta_path = output_path.parent / f"{output_path.stem}_metadata.json"
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "source_role_assignments": str(self.role_assignments_path),
            "source_spider_data": str(self.spider_data_path),
            "seed": self.seed,
            "config": {
                "roles_per_db": roles_per_db,
                "queries_per_role": queries_per_role,
                "balance_ratio": balance_ratio
            },
            "statistics": {
                "total_entries": len(entries),
                "allowed": allowed_count,
                "denied": denied_count,
                "allowed_ratio": allowed_count / len(entries) if entries else 0,
                "databases": len(by_db),
                "by_database": by_db
            }
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset saved to {output_path}")
        print(f"Metadata saved to {meta_path}")
        print(f"  Total entries: {len(entries)}")
        print(f"  Allowed: {allowed_count} ({allowed_count/len(entries)*100:.1f}%)" if entries else "  No entries")
        print(f"  Denied: {denied_count} ({denied_count/len(entries)*100:.1f}%)" if entries else "")
        print(f"  Databases: {len(by_db)}")
        
        return metadata["statistics"]


def generate_spider_column_level_dataset(
    role_assignments_path: str,
    spider_data_path: str,
    spider_db_root: str,
    output_path: str,
    project_root: Optional[str] = None,
    roles_per_db: Optional[int] = None,
    queries_per_role: Optional[int] = None,
    seed: int = 42
) -> Dict:
    """
    Convenience function to generate Spider column-level RBAC dataset.
    
    Args:
        role_assignments_path: Path to role assignments JSON
        spider_data_path: Path to Spider dev.json/test.json
        spider_db_root: Root of Spider databases
        output_path: Output JSON path
        project_root: Project root (optional)
        roles_per_db: Roles to use per database (None = all)
        queries_per_role: Queries per role (None = all)
        seed: Random seed
    
    Returns:
        Statistics dictionary
    """
    generator = SpiderColumnLevelRBACGenerator(
        role_assignments_path=role_assignments_path,
        spider_data_path=spider_data_path,
        spider_db_root=spider_db_root,
        project_root=project_root,
        seed=seed
    )
    
    return generator.generate_and_save(
        output_path=output_path,
        roles_per_db=roles_per_db,
        queries_per_role=queries_per_role
    )
