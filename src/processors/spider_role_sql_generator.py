"""Generate role-based access control text2sql dataset for Spider."""

import json
import sys
import logging
import sqlite3
import sqlparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from nltk import word_tokenize

ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger('role_sql')

CLAUSE_KEYWORDS = (
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "intersect",
    "union",
    "except",
)
JOIN_KEYWORDS = ("join", "on", "as")

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}

COND_OPS = ("and", "or")
SQL_OPS = ("intersect", "union", "except")
ORDER_OPS = ("desc", "asc")


class Schema:
    """Simple schema mapping table & column names to unique identifiers."""

    def __init__(self, schema: Dict[str, List[str]]):
        self._schema = schema
        self._id_map = self._create_id_map(schema)

    @property
    def schema(self) -> Dict[str, List[str]]:
        return self._schema

    @property
    def idMap(self) -> Dict[str, str]:  # Retain legacy attribute name for compatibility
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


def get_schema(db_path: str) -> Dict[str, List[str]]:
    """Load table → column mapping from a Spider SQLite database."""

    schema: Dict[str, List[str]] = {}
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [str(result[0].lower()) for result in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]
    finally:
        conn.close()

    return schema


def tokenize(sql: str) -> List[str]:
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
    alias_map: Dict[str, str] = {}
    as_positions = [idx for idx, tok in enumerate(tokens) if tok == "as"]
    for idx in as_positions:
        alias_map[tokens[idx + 1]] = tokens[idx - 1]
    return alias_map


def get_tables_with_alias(schema: Dict[str, List[str]], tokens: List[str]) -> Dict[str, str]:
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
    return unit[0] != AGG_OPS.index("none")


def parse_val_unit(
    tokens: List[str],
    start_idx: int,
    tables_with_alias: Dict[str, str],
    schema: Schema,
    default_tables: Optional[List[str]] = None,
) -> Tuple[int, Tuple[int, Tuple[int, str, bool], Optional[Tuple[int, str, bool]]]]:
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
) -> Tuple[int, Tuple[bool, List[Tuple[int, Tuple[int, Tuple[int, str, bool], Optional[Tuple[int, str, bool]]]]]]]:
    idx = start_idx
    length = len(tokens)

    assert tokens[idx] == "select"
    idx += 1
    is_distinct = False
    if idx < length and tokens[idx] == "distinct":
        idx += 1
        is_distinct = True

    val_units: List[Tuple[int, Tuple[int, Tuple[int, str, bool], Optional[Tuple[int, str, bool]]]]] = []

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
) -> Tuple[int, Tuple[str, List[Tuple[int, Tuple[int, str, bool], Optional[Tuple[int, str, bool]]]]]]:
    idx = start_idx
    length = len(tokens)
    val_units: List[Tuple[int, Tuple[int, str, bool], Optional[Tuple[int, str, bool]]]] = []
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
    idx = start_idx
    length = len(tokens)

    if idx >= length or tokens[idx] != "having":
        return idx, []

    idx += 1
    idx, conditions = parse_condition(tokens, idx, tables_with_alias, schema, default_tables)
    return idx, conditions


def parse_limit(tokens: List[str], start_idx: int) -> Tuple[int, Optional[int]]:
    idx = start_idx
    length = len(tokens)

    if idx < length and tokens[idx] == "limit":
        idx += 2
        if not isinstance(tokens[idx - 1], int):
            return idx, 1
        return idx, int(tokens[idx - 1])

    return idx, None


def skip_semicolon(tokens: List[str], start_idx: int) -> int:
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
    tokens = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, tokens)
    _, sql = parse_sql(tokens, 0, tables_with_alias, schema)
    return sql


def get_nestedSQL(sql: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql: Dict[str, Any]) -> int:
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
    return len(get_nestedSQL(sql))


def count_others(sql: Dict[str, Any]) -> int:
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
    """Compute Spider difficulty category from parsed SQL dictionary."""

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

class RoleSQLGenerator:
    """Generate role-based SQL queries dataset based on access control."""

    _latest_baseline: List[Dict[str, Any]]
    
    def __init__(self, project_root: Path, output_dir: str = None):
        """
        Initialize generator with project root path and optional output directory.
        
        Args:
            project_root (Path): Path to project root directory
            output_dir (str, optional): Path to output directory for generated files
        """
        self.project_root = project_root
        self.data_dir = project_root / 'data'
        self.output_dir = Path(output_dir) if output_dir else project_root / 'outputs'
        
        # Data source mapping
        self.data_sources = {
            'train': self.data_dir / 'spider_train_data.json',
            'dev': self.data_dir / 'spider_dev_data.json', 
            'test': self.data_dir / 'spider_test_data.json',
            'combined': self.data_dir / 'spider_combined_data.json'
        }
        
        # Tables information files
        self.tables_files = {
            'train_dev': project_root / 'data' / 'spider' / 'tables.json',
            'test': project_root / 'data' / 'spider' / 'test_tables.json'
        }

        # Cache for table information, schema, instructions, and difficulties
        self._tables_cache: Dict[str, Dict[str, Any]] = {}
        self._schema_cache: Dict[str, Schema] = {}
        self._difficulty_cache: Dict[Tuple[str, str], Optional[str]] = {}
        self._instruction_cache: Dict[str, str] = {}
        self._latest_baseline = []
        
    def extract_tables_from_query(self, query: str) -> Set[str]:
        """
        Extract table names from a SQL query.
        
        Args:
            query (str): SQL query string
            
        Returns:
            Set[str]: Set of table names used in the query
        """
        tables = set()
        
        try:
            # Parse the SQL query
            statements = sqlparse.parse(query)
            
            for statement in statements:
                # Use a more targeted approach to find table names
                self._extract_tables_from_statement(statement, tables)
                        
            # Log extraction results
            if tables:
                logger.debug(f"Extracted tables from query: {tables}")
            else:
                logger.warning(f"No tables extracted from query: {query}")
                
        except Exception as e:
            logger.error(f"Error extracting tables from query: {e}")
            logger.error(f"Query: {query}")
            
        return tables
    
    def _extract_tables_from_statement(self, statement, tables):
        """Extract table names from a parsed SQL statement"""
        # Convert statement to string and use regex-based approach for better accuracy
        statement_str = str(statement).strip()
        
        # Use regex to find FROM and JOIN clauses
        import re
        
        # Pattern to match FROM clause: FROM table_name or FROM table AS alias
        from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?)'
        from_matches = re.finditer(from_pattern, statement_str, re.IGNORECASE)
        
        for match in from_matches:
            table_part = match.group(1).strip()
            table_name = self._clean_table_name(table_part)
            if table_name:
                tables.add(table_name)
                logger.debug(f"Extracted table name from FROM: {table_name}")
        
        # Pattern to match JOIN clauses: JOIN table_name or JOIN table AS alias
        join_pattern = r'\b(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+|OUTER\s+)?JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?)'
        join_matches = re.finditer(join_pattern, statement_str, re.IGNORECASE)
        
        for match in join_matches:
            table_part = match.group(1).strip()
            table_name = self._clean_table_name(table_part)
            if table_name:
                tables.add(table_name)
                logger.debug(f"Extracted table name from JOIN: {table_name}")
        
        # Handle subqueries
        subquery_pattern = r'\(([^()]*SELECT[^()]*)\)'
        subqueries = re.findall(subquery_pattern, statement_str, re.IGNORECASE | re.DOTALL)
        
        for subquery in subqueries:
            subquery_tables = self.extract_tables_from_query(subquery.strip())
            tables.update(subquery_tables)
    
    def _clean_table_name(self, name: str) -> str:
        """Clean and normalize table name"""
        if not name:
            return None
            
        name = name.strip()
        
        # Handle alias cases - split on space and take first part
        if ' ' in name:
            parts = name.split()
            # Check if it's an alias pattern (table_name alias_name)
            if len(parts) >= 2 and parts[1].upper() not in ('AS', 'ON', 'WHERE', 'GROUP', 'ORDER', 'HAVING'):
                name = parts[0]
            else:
                name = parts[0]
        
        # Handle explicit AS aliases
        alias_markers = [' as ', ' AS ', ' As ', ' aS ']
        for marker in alias_markers:
            if marker in name:
                name = name.split(marker)[0]
                break
                
        # Handle schema qualification
        if '.' in name:
            name = name.split('.')[-1]
            
        # Clean up any remaining special characters
        name = name.strip('"`[]() \n\t\r')
        
        # Validate and return - exclude SQL keywords
        excluded_keywords = {
            'select', 'from', 'where', 'group', 'order', 'having', 'join', 'on', 'as',
            'inner', 'left', 'right', 'full', 'outer', 'union', 'intersect', 'except',
            'limit', 'offset', 'distinct', 'all', 'and', 'or', 'not', 'in', 'exists',
            'between', 'like', 'is', 'null', 'true', 'false', 'case', 'when', 'then',
            'else', 'end', 'count', 'sum', 'avg', 'min', 'max'
        }
        
        if name and not name.isspace() and name.lower() not in excluded_keywords:
            return name.lower()
        return None
    
    def _extract_from_complex_token(self, token, tables):
        """Helper method to extract table names from complex tokens"""
        if isinstance(token, sqlparse.sql.Identifier):
            # Handle table identifiers
            table_name = token.get_real_name()
            if table_name:
                clean_name = table_name.strip('"`[]() \n\t\r').lower()
                if clean_name and clean_name not in ('select', 'from', 'where', 'group', 'order', 'having', 'join', 'on', 'as'):
                    tables.add(clean_name)
                    logger.debug(f"Extracted table name from identifier: {clean_name}")
        
        elif isinstance(token, sqlparse.sql.IdentifierList):
            # Handle lists of identifiers
            for identifier in token.get_identifiers():
                table_name = identifier.get_real_name() if hasattr(identifier, 'get_real_name') else str(identifier)
                if table_name:
                    clean_name = table_name.strip('"`[]() \n\t\r').lower()
                    if clean_name and clean_name not in ('select', 'from', 'where', 'group', 'order', 'having', 'join', 'on', 'as'):
                        tables.add(clean_name)
                        logger.debug(f"Extracted table name from identifier list: {clean_name}")
    
    def _extract_from_subqueries(self, query, tables):
        """Helper method to extract table names from subqueries"""
        try:
            # Find subqueries in parentheses
            import re
            subquery_pattern = r'\(([^()]*SELECT[^()]*)\)'
            subqueries = re.findall(subquery_pattern, query, re.IGNORECASE | re.DOTALL)
            
            for subquery in subqueries:
                subquery_tables = self.extract_tables_from_query(subquery.strip())
                tables.update(subquery_tables)
                
        except Exception as e:
            logger.debug(f"Error extracting from subqueries: {e}")
            
        return tables
        
    def load_role_assignments(self, role_file_path: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load role assignments from the specified file path or latest file.
        
        Args:
            role_file_path (str, optional): Path to role assignments JSON file
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Role assignments by database
        """
        try:
            if role_file_path:
                role_file = Path(role_file_path)
            else:
                # Get latest role assignment file
                role_files = list(self.output_dir.glob('role_assignments_*.json'))
                if not role_files:
                    raise FileNotFoundError("No role assignment files found")
                role_file = max(role_files, key=lambda x: x.stat().st_mtime)
                
            with open(role_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data['assignments']
                
        except Exception as e:
            logger.error(f"Error loading role assignments: {str(e)}")
            raise
            
    def load_spider_data(self, data_source: str = 'train') -> List[Dict[str, str]]:
        """
        Load Spider data from specified data source.
        
        Args:
            data_source (str): Data source type ('train', 'dev', 'test', 'combined')
        
        Returns:
            List[Dict[str, str]]: List of Spider examples
        """
        if data_source not in self.data_sources:
            raise ValueError(f"Unknown data source: {data_source}")
            
        data_path = self.data_sources[data_source]
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return []
            
    def load_table_info(self, data_source: str = 'train') -> Dict[str, Dict[str, Any]]:
        """
        Load table information from Spider tables.json files.
        
        Args:
            data_source (str): Data source type ('train', 'dev', 'test', 'combined')
            
        Returns:
            Dict[str, Dict[str, Any]]: Table information by database ID
        """
        # Determine which tables file to use
        if data_source in ['train', 'dev', 'combined']:
            tables_key = 'train_dev'
        else:  # test
            tables_key = 'test'
            
        # Check cache first
        if tables_key in self._tables_cache:
            return self._tables_cache[tables_key]
            
        tables_file = self.tables_files[tables_key]
        
        try:
            with open(tables_file, 'r', encoding='utf-8') as f:
                tables_list = json.load(f)
            
            # Convert to dict indexed by db_id
            tables_dict = {item['db_id']: item for item in tables_list}
            
            # Cache the result
            self._tables_cache[tables_key] = tables_dict
            
            logger.info(f"Loaded table info for {len(tables_dict)} databases from {tables_file}")
            return tables_dict
            
        except FileNotFoundError:
            logger.error(f"Tables file not found: {tables_file}")
            return {}
        except Exception as e:
            logger.error(f"Error loading table info: {str(e)}")
            return {}
    
    def generate_database_instruction(self, db_id: str, table_info: Dict[str, Any]) -> str:
        """
        Generate database instruction similar to decode_json_file function.
        
        Args:
            db_id (str): Database ID
            table_info (Dict[str, Any]): Table information from tables.json
            
        Returns:
            str: Generated instruction string
        """
        try:
            tables = table_info["table_names_original"]
            columns = table_info["column_names_original"][1:]  # Skip the first [-1, "*"] entry
            primary_keys = table_info["primary_keys"]
            foreign_keys = table_info["foreign_keys"]
            
            # Start building the instruction
            instruction = f"{db_id} contains tables such as " + ", ".join(tables) + ". "
            
            # Add column information for each table
            for i, table_name in enumerate(tables):
                table_columns = [column[1] for column in columns if column[0] == i]
                instruction += f"Table {table_name} has columns such as " + ", ".join(table_columns) + ". "
                
                # Add primary key information
                for j in range(len(primary_keys)):
                    if isinstance(primary_keys[j], int):
                        # Single primary key
                        if columns[primary_keys[j] - 1][0] == i:
                            instruction += f"{columns[primary_keys[j] - 1][1]} is the primary key.\n"
                    elif isinstance(primary_keys[j], list):
                        # Composite primary key
                        keys = []
                        for k in range(len(primary_keys[j])):
                            if columns[primary_keys[j][k] - 1][0] == i:
                                keys.append(columns[primary_keys[j][k] - 1][1])
                        if keys:
                            instruction += f"The combination of ({', '.join(keys)}) are the primary key.\n"
            
            # Add foreign key information
            for key in foreign_keys:
                try:
                    foreign_column = columns[key[0] - 1][1]
                    foreign_table = tables[columns[key[0] - 1][0]]
                    reference_column = columns[key[1] - 1][1]
                    reference_table = tables[columns[key[1] - 1][0]]
                    
                    instruction += f"The {foreign_column} of {foreign_table} is the foreign key of {reference_column} of {reference_table}.\n"
                except (IndexError, KeyError) as e:
                    logger.debug(f"Skipping invalid foreign key in {db_id}: {key}, error: {str(e)}")
                    continue
            
            return instruction.strip()
            
        except Exception as e:
            logger.error(f"Error generating instruction for {db_id}: {str(e)}")
            return f"{db_id} database schema information."

    def _resolve_sqlite_path(self, db_id: str, data_source: str) -> Optional[Path]:
        """Resolve the path to the Spider SQLite database for a given db_id."""
        base_dir = self.project_root / 'data' / 'spider'
        candidates: List[Path] = []

        if data_source == 'test':
            candidates.append(base_dir / 'test_database' / db_id / f"{db_id}.sqlite")
            candidates.append(base_dir / 'database' / db_id / f"{db_id}.sqlite")
        else:
            candidates.append(base_dir / 'database' / db_id / f"{db_id}.sqlite")
            candidates.append(base_dir / 'test_database' / db_id / f"{db_id}.sqlite")

        for path in candidates:
            if path.exists():
                return path

        logger.debug("Could not locate SQLite file for %s (candidates=%s)", db_id, candidates)
        return None

    def _load_schema(self, db_id: str, data_source: str) -> Optional[Schema]:
        """Load and cache the parsed schema for a Spider database."""
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]

        db_path = self._resolve_sqlite_path(db_id, data_source)
        if db_path is None:
            logger.warning("SQLite database not found for %s (source=%s)", db_id, data_source)
            return None

        try:
            schema = Schema(get_schema(str(db_path)))
            self._schema_cache[db_id] = schema
            return schema
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to load schema for %s: %s", db_id, exc)
            return None

    def _compute_sql_difficulty(self, db_id: str, sql_query: str, data_source: str) -> Optional[str]:
        """Compute Spider difficulty level for a SQL query."""
        normalized_sql = " ".join(sql_query.split())
        cache_key = (db_id, normalized_sql)
        if cache_key in self._difficulty_cache:
            return self._difficulty_cache[cache_key]

        schema = self._load_schema(db_id, data_source)
        if schema is None:
            self._difficulty_cache[cache_key] = None
            return None

        try:
            parsed_sql = get_sql(schema, normalized_sql)
            difficulty = compute_spider_difficulty(parsed_sql)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Could not parse SQL for difficulty (db=%s): %s", db_id, exc)
            difficulty = None

        self._difficulty_cache[cache_key] = difficulty
        return difficulty

    def _prepare_example_metadata(
        self,
        example: Dict[str, Any],
        instruction_template: str,
        table_info: Dict[str, Dict[str, Any]],
        data_source: str,
    ) -> Optional[Dict[str, Any]]:
        """Prepare shared metadata fields for role and baseline outputs."""

        db_id = example.get('db_id')
        if not db_id:
            logger.debug("Skipping record without db_id: %s", example)
            return None

        question = example.get('question') or example.get('input')
        sql_query = example.get('query') or example.get('output')
        if question is None or not sql_query:
            logger.debug(
                "Skipping record for %s due to missing question or SQL (question=%s, sql_present=%s)",
                db_id,
                question is not None,
                bool(sql_query),
            )
            return None

        instruction = example.get('instruction')
        if not instruction:
            if db_id in self._instruction_cache:
                instruction = self._instruction_cache[db_id]
            elif db_id in table_info:
                db_instruction = self.generate_database_instruction(db_id, table_info[db_id])
                instruction = instruction_template.format(db_instruction)
                self._instruction_cache[db_id] = instruction
            else:
                logger.warning("No table info found for database: %s", db_id)
                instruction = instruction_template.format(f"{db_id} database")
                self._instruction_cache[db_id] = instruction
        else:
            self._instruction_cache.setdefault(db_id, instruction)

        input_text = example.get('input') or question
        difficulty = self._compute_sql_difficulty(db_id, sql_query, data_source) or "unknown"

        return {
            'db_id': db_id,
            'instruction': instruction,
            'input': input_text,
            'gold_sql': sql_query,
            'difficulty': difficulty,
        }
            
    def check_query_permission(self, query_tables: Set[str], role_tables: str) -> bool:
        """
        Check if a role has permission to access all tables in a query.
        
        Args:
            query_tables (Set[str]): Tables used in the query
            role_tables (str): Tables accessible by the role
            
        Returns:
            bool: True if role has access to all required tables
        """
        # Convert role_tables string to set and normalize to lower case
        role_tables_set = {t.strip().lower() for t in role_tables.split(',') if t.strip()}
        
        # Convert query tables to lower case
        query_tables_lower = {t.lower() for t in query_tables if t.strip()}
        
        # Log the permission check details
        logger.debug(f"Checking permissions:")
        logger.debug(f"Role has access to: {role_tables_set}")
        logger.debug(f"Query uses tables: {query_tables_lower}")
        
        # Check each required table
        for table in query_tables_lower:
            if table not in role_tables_set:
                logger.info(f"Permission denied: Table '{table}' is not accessible")
                return False
                
        return True
        
    def generate_role_sql_dataset(
        self,
        role_file_path: str = None,
        data_source: str = 'train',
        *,
        spider_data: Optional[List[Dict[str, Any]]] = None,
        instruction_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Generate Spider role-conditioned dataset with minimal fields.

        Args:
            role_file_path: Path to the role assignment JSON file.
            data_source: Spider split to use (train, dev, test, combined).
            spider_data: Optional pre-processed examples; if omitted, the split
                will be loaded from disk.
            instruction_prompt: Optional override for instruction template. When
                omitted, the prompt from ``configs/prompts.py`` (``INSTRUCTION_PROMPT``)
                is used.

        Returns:
            A list of dictionaries. Each entry contains ``db_id``, ``instruction``,
            ``role``, ``tables``, ``input``, ``output``, ``gold_sql``, and ``difficulty``.
        """

        role_assignments = self.load_role_assignments(role_file_path)
        if not role_assignments:
            logger.error("No role assignments available; aborting dataset generation.")
            return []

        if spider_data is None:
            spider_data = self.load_spider_data(data_source)

        if not spider_data:
            logger.error("Spider source data is empty; aborting dataset generation.")
            return []

        table_info = self.load_table_info(data_source)

        baseline_entries: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        try:
            configs_path = self.project_root / 'configs'
            if str(configs_path) not in sys.path:
                sys.path.insert(0, str(configs_path))
            from prompts import INSTRUCTION_PROMPT  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("Could not import INSTRUCTION_PROMPT from configs; using default template.")
            INSTRUCTION_PROMPT = (
                "I want you to act as a SQL terminal in front of an example database, you need only to "
                "return the sql command to me.Below is an instruction that describes a task, Write a response "
                "that appropriately completes the request.\n##Instruction:\n{}\n"
            )

        instruction_template = instruction_prompt or INSTRUCTION_PROMPT

        dataset: List[Dict[str, Any]] = []
        processed_dbs: Set[str] = set()
        skipped_dbs: Set[str] = set()
        allowed_count = 0
        denied_count = 0

        for example in spider_data:
            db_id = example.get('db_id')
            if not db_id:
                logger.debug("Skipping record without db_id: %s", example)
                continue

            if db_id not in role_assignments:
                skipped_dbs.add(db_id)
                continue

            base_entry = self._prepare_example_metadata(example, instruction_template, table_info, data_source)
            if base_entry is None:
                continue

            sql_query = base_entry['gold_sql']
            processed_dbs.add(db_id)

            baseline_key = (base_entry['db_id'], base_entry['input'], base_entry['gold_sql'])
            if baseline_key not in baseline_entries:
                baseline_entries[baseline_key] = {
                    'db_id': base_entry['db_id'],
                    'instruction': base_entry['instruction'],
                    'input': base_entry['input'],
                    'output': base_entry['gold_sql'],
                    'gold_sql': base_entry['gold_sql'],
                    'difficulty': base_entry['difficulty'],
                }

            query_tables = self.extract_tables_from_query(sql_query)

            for role_info in role_assignments[db_id]:
                role_name = role_info.get('role')
                role_tables = role_info.get('tables', '')

                if not role_name:
                    logger.debug("Skipping role entry without name for database %s", db_id)
                    continue

                has_permission = self.check_query_permission(query_tables, role_tables)
                output_sql = sql_query if has_permission else "Sorry, I cannot answer."

                if has_permission:
                    allowed_count += 1
                else:
                    denied_count += 1

                entry = {
                    'db_id': base_entry['db_id'],
                    'instruction': base_entry['instruction'],
                    'role': role_name,
                    'tables': role_tables,
                    'input': base_entry['input'],
                    'output': output_sql,
                    'gold_sql': base_entry['gold_sql'],
                    'difficulty': base_entry['difficulty'],
                }

                dataset.append(entry)

        logger.info("Dataset generation completed for %s data:", data_source)
        logger.info("- Databases processed: %s", len(processed_dbs))
        logger.info("- Databases skipped (no roles): %s", len(skipped_dbs))
        logger.info("- Total examples generated: %s", len(dataset))
        logger.info("- Allowed queries: %s", allowed_count)
        logger.info("- Denied queries: %s", denied_count)

        if skipped_dbs:
            logger.debug("Skipped databases: %s", sorted(skipped_dbs))

        self._latest_baseline = list(baseline_entries.values())

        return dataset

    def generate_baseline_dataset(
        self,
        data_source: str = 'train',
        *,
        spider_data: Optional[List[Dict[str, Any]]] = None,
        instruction_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate Spider dataset without role conditioning using minimal fields.

        Returns:
            List[Dict[str, Any]]: Entries include ``db_id``, ``instruction``,
            ``input``, ``output``, and ``difficulty``.
        """

        if spider_data is None:
            spider_data = self.load_spider_data(data_source)
        if not spider_data:
            logger.error("Spider source data is empty; aborting baseline generation.")
            return []

        table_info = self.load_table_info(data_source)

        try:
            configs_path = self.project_root / 'configs'
            if str(configs_path) not in sys.path:
                sys.path.insert(0, str(configs_path))
            from prompts import INSTRUCTION_PROMPT  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("Could not import INSTRUCTION_PROMPT from configs; using default template.")
            INSTRUCTION_PROMPT = (
                "I want you to act as a SQL terminal in front of an example database, you need only to "
                "return the sql command to me.Below is an instruction that describes a task, Write a response "
                "that appropriately completes the request.\n##Instruction:\n{}\n"
            )

        instruction_template = instruction_prompt or INSTRUCTION_PROMPT

        baseline: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[str, str, str]] = set()

        for example in spider_data:
            base_entry = self._prepare_example_metadata(example, instruction_template, table_info, data_source)
            if base_entry is None:
                continue

            key = (base_entry['db_id'], base_entry['input'], base_entry['gold_sql'])
            if key in seen_keys:
                continue
            seen_keys.add(key)

            baseline.append(
                {
                    'db_id': base_entry['db_id'],
                    'instruction': base_entry['instruction'],
                    'input': base_entry['input'],
                    'output': base_entry['gold_sql'],
                    'difficulty': base_entry['difficulty'],
                }
            )

        logger.info(
            "Baseline dataset generation completed for %s data: total examples %s (unique by db/input/sql)",
            data_source,
            len(baseline),
        )

        return baseline
        
    def generate_all_datasets(self, role_file_path: str = None, timestamp: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate role-based SQL datasets for all available data sources.
        
        Args:
            role_file_path (str, optional): Path to role assignments JSON file
            timestamp (str, optional): Timestamp for output filenames
            
        Returns:
            Dict[str, Dict[str, Any]]: Results for each data source with statistics
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {}
        
        # Process each data source
        for data_source in ['train', 'dev', 'test']:
            try:
                # Check if data file exists
                if not self.data_sources[data_source].exists():
                    logger.warning(f"Data file for {data_source} not found, skipping")
                    continue
                
                logger.info(f"\nGenerating {data_source} dataset...")
                
                # Generate dataset
                dataset = self.generate_role_sql_dataset(role_file_path, data_source)
                
                if dataset:
                    # Save dataset
                    output_path = self.output_dir / f'role_sql_dataset_{data_source}_{timestamp}.json'
                    self.save_dataset(dataset, str(output_path))
                    
                    # Calculate statistics
                    total_examples = len(dataset)
                    databases = len({example['db_id'] for example in dataset})
                    roles = len({(example['db_id'], example['role']) for example in dataset})
                    denied_queries = sum(1 for example in dataset if example.get('output') == "Sorry, I cannot answer.")
                    
                    results[data_source] = {
                        'total_examples': total_examples,
                        'databases': databases,
                        'roles': roles,
                        'denied_queries': denied_queries,
                        'denial_rate': denied_queries/total_examples*100 if total_examples else 0,
                        'output_path': str(output_path)
                    }
                    
                    logger.info(f"{data_source.title()} dataset statistics:")
                    logger.info(f"- Total examples: {total_examples}")
                    logger.info(f"- Databases: {databases}")
                    logger.info(f"- Unique roles: {roles}")
                    logger.info(f"- Denied queries: {denied_queries} ({denied_queries/total_examples*100:.2f}%)")
                else:
                    logger.warning(f"No dataset generated for {data_source}")
                    results[data_source] = {'error': 'No dataset generated'}
                    
            except Exception as e:
                logger.error(f"Failed to process {data_source} dataset: {str(e)}")
                results[data_source] = {'error': str(e)}
        
        # Generate combined dataset if individual ones exist
        if any('error' not in result for result in results.values()):
            try:
                # Check if combined data file exists
                if self.data_sources['combined'].exists():
                    logger.info(f"\nGenerating combined dataset...")
                    combined_dataset = self.generate_role_sql_dataset(role_file_path, 'combined')
                    
                    if combined_dataset:
                        output_path = self.output_dir / f'role_sql_dataset_combined_{timestamp}.json'
                        self.save_dataset(combined_dataset, str(output_path))
                        
                        total_examples = len(combined_dataset)
                        databases = len({example['db_id'] for example in combined_dataset})
                        roles = len({(example['db_id'], example['role']) for example in combined_dataset})
                        denied_queries = sum(1 for example in combined_dataset if example.get('output') == "Sorry, I cannot answer.")
                        
                        results['combined'] = {
                            'total_examples': total_examples,
                            'databases': databases,
                            'roles': roles,
                            'denied_queries': denied_queries,
                            'denial_rate': denied_queries/total_examples*100 if total_examples else 0,
                            'output_path': str(output_path)
                        }
                        
                        logger.info(f"Combined dataset statistics:")
                        logger.info(f"- Total examples: {total_examples}")
                        logger.info(f"- Databases: {databases}")
                        logger.info(f"- Unique roles: {roles}")
                        logger.info(f"- Denied queries: {denied_queries} ({denied_queries/total_examples*100:.2f}%)")
                
            except Exception as e:
                logger.error(f"Failed to generate combined dataset: {str(e)}")
                results['combined'] = {'error': str(e)}
        
        return results
        
    def save_dataset(self, dataset: List[Dict[str, str]], output_path: str = None) -> None:
        """
        Save generated dataset to file.
        
        Args:
            dataset (List[Dict[str, str]]): Generated dataset
            output_path (str, optional): Path for output file. If not provided,
                                       will save with timestamp in output directory
        """
        try:
            if output_path:
                output_file = Path(output_path)
                # Create parent directories if needed
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Create output directory if needed
                self.output_dir.mkdir(parents=True, exist_ok=True)
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = self.output_dir / f'role_sql_dataset_{timestamp}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Dataset saved to {output_file}")
            logger.info(f"Total examples generated: {len(dataset)}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            raise
            
if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get project root directory
    project_root = Path(__file__).parents[2]
    
    # Initialize generator
    generator = RoleSQLGenerator(
        project_root=project_root,
        output_dir=project_root / 'outputs'
    )
    
    # Example: Generate datasets for all data sources
    role_file = project_root / 'outputs/role_assignments_20250917_111537.json'
    results = generator.generate_all_datasets(role_file_path=str(role_file))
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET GENERATION SUMMARY")
    print("="*50)
    
    for data_source, result in results.items():
        print(f"\n{data_source.upper()} Dataset:")
        if 'error' in result:
            print(f"   Error: {result['error']}")
        else:
            print(f"   Examples: {result['total_examples']}")
            print(f"   Databases: {result['databases']}")
            print(f"   Roles: {result['roles']}")
            print(f"   Denied: {result['denied_queries']} ({result['denial_rate']:.1f}%)")
            print(f"   Output: {result['output_path']}")
    
    print("\n" + "="*50)
