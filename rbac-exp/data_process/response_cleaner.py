"""
Response cleaning utilities for RBAC benchmark evaluation.
Aligned with experiments/data_process/cloud_output_cleaner.py and
experiments/predict/response_cleaner.py for consistency.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple, Match

# =============================================================================
# Constants (aligned with experiments)
# =============================================================================

SQL_KEYWORDS = (
    "SELECT",
    "WITH",
    "INSERT",
    "UPDATE",
    "DELETE",
    "CREATE",
    "DROP",
    "ALTER",
    "DESCRIBE",
    "DESC",
    "SHOW",
    "USE",
    "SET",
    "EXPLAIN",
)

EXPLANATION_MARKERS = (
    "note:",
    "notes:",
    "explanation:",
    "this query",
    "this command",
    "this sql",
    "this statement",
    "this will",
    "these queries",
    "it returns",
    "it will",
    "here is",
    "here are",
)

REFUSAL_PATTERNS = (
    "sorry",
    "cannot answer",
    "can't answer",
    "cannot provide",
    "can't provide",
    "cannot assist",
    "can't assist",
    "cannot help",
    "can't help",
    "unable to answer",
    "unable to provide",
    "unable to assist",
    "unable to help",
    "empty response",
    "not enough information",
    "no information",
    "the provided schema",
    "there is no column",
    "there is no information",
    "it seems that the information",
    "cannot be determined",
    "don't have permission",
    "do not have permission",
    "don't have access",
    "do not have access",
    "not authorized",
    "access denied",
    "permission denied",
    "insufficient permission",
    "no permission",
    "i apologize",
    "i'm sorry",
    "i am sorry",
)

# Code block patterns (from experiments/predict/response_cleaner.py)
CODE_BLOCK_RE = re.compile(
    r"```+\s*(?P<lang>sql|postgresql|sqlite|mysql|snowflake|text)?\s*(?P<code>[\s\S]*?)```+",
    re.IGNORECASE,
)
SQL_LANG_ALIASES = {"sql", "postgresql", "sqlite", "mysql", "snowflake"}
SQL_PREFIX_RE = re.compile(
    r"^(?:sql\s*query|sql|postgresql|sqlite|mysql|query|answer|analysis|statement|output)\s*[:：\-]*",
    re.IGNORECASE,
)
LEADING_BULLET_RE = re.compile(r"^\s*([*\-•]+\s+)")


# =============================================================================
# Low-level cleaning functions (from experiments/predict/response_cleaner.py)
# =============================================================================

def _select_default_candidate(text: str) -> str:
    """Return the SQL snippet using the legacy selection strategy."""
    sql_candidate: str | None = None
    other_candidate: str | None = None
    for match in CODE_BLOCK_RE.finditer(text):
        block = (match.group("code") or "").strip()
        if not block:
            continue
        lang = (match.group("lang") or "").lower()
        if lang in SQL_LANG_ALIASES:
            sql_candidate = block
        else:
            other_candidate = block

    if sql_candidate:
        return sql_candidate
    if other_candidate:
        return other_candidate
    return strip_code_fences(text)


def _extract_last_code_block(text: str, *, allowed_langs: set[str] | None) -> str | None:
    """Return the last fenced code block whose language is in allowed_langs."""
    last_block: str | None = None
    for match in CODE_BLOCK_RE.finditer(text):
        code = (match.group("code") or "").strip()
        if not code:
            continue
        if allowed_langs is not None:
            lang = (match.group("lang") or "").lower()
            if lang not in allowed_langs:
                continue
        last_block = code
    return last_block


def extract_last_sql_code_block(text: str) -> str | None:
    """Return the final fenced SQL code block within text if present."""
    return _extract_last_code_block(text, allowed_langs=SQL_LANG_ALIASES)


def strip_code_fences(text: str) -> str:
    """Remove fenced code blocks while preserving the enclosed SQL."""
    def _replace(match: Match[str]) -> str:
        inner = match.group("code") or ""
        return inner.strip()
    return CODE_BLOCK_RE.sub(_replace, text)


def clean_model_response(response: str, *, mode: str = "default") -> str:
    """
    Normalize raw LLM output to a compact SQL statement.
    (Aligned with experiments/predict/response_cleaner.py)
    
    Args:
        response: Raw model output text.
        mode: Cleaning mode. "default" preserves legacy behaviour, while
            "snowflake" enables heuristics tailored for Snowflake outputs.
    
    Returns:
        Cleaned SQL string, "Sorry, I cannot answer." for refusals, 
        or "empty response" for invalid/empty output.
    """
    if not response:
        return "empty response"

    text = response.strip()
    if not text:
        return "empty response"
    
    if text.startswith("Error:"):
        return text

    mode_key = mode.lower()
    if mode_key not in {"default", "snowflake"}:
        raise ValueError(f"Unsupported clean_model_response mode: {mode!r}")

    if mode_key == "snowflake":
        sql_candidate = extract_last_sql_code_block(text)
        if sql_candidate:
            candidate = sql_candidate
        else:
            other_candidate = _extract_last_code_block(text, allowed_langs=None)
            candidate = other_candidate if other_candidate else strip_code_fences(text)
    else:
        candidate = _select_default_candidate(text)

    candidate = strip_code_fences(candidate)
    candidate = candidate.replace("\r\n", "\n")

    lines = []
    for raw_line in candidate.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("```") or line.endswith("```"):
            line = line.strip("`")
        line = SQL_PREFIX_RE.sub("", line).strip()
        if not line:
            continue
        line = LEADING_BULLET_RE.sub("", line).strip()
        if not line:
            continue
        lines.append(line)

    cleaned = " ".join(lines).strip()
    cleaned = re.sub(r"`{3,}", "", cleaned)

    # Handle llama-3-sqlcoder issue: remove standalone "assistant" word and everything after
    # This fixes cases like: "SELECT * FROM t;assistant I am looking for..."
    # Match "assistant" that is not part of an identifier (e.g., not "teaching_assistant")
    assistant_match = re.search(r'(?<![a-zA-Z_])assistant\b', cleaned, re.IGNORECASE)
    if assistant_match:
        cleaned = cleaned[:assistant_match.start()].strip()
    
    # Validate that the result looks like SQL - must start with a SQL keyword
    keyword_regex = re.compile(r"^\s*(" + "|".join(SQL_KEYWORDS) + r")\b", re.IGNORECASE)
    if cleaned and keyword_regex.match(cleaned):
        # Valid SQL found - return it
        return cleaned.strip()
    
    # No valid SQL found - check if this is a refusal response
    lowered = text.lower()
    if any(pattern in lowered for pattern in REFUSAL_PATTERNS):
        return "Sorry, I cannot answer."
    
    # Not valid SQL and not a recognized refusal - return "empty response"
    return "empty response"


# =============================================================================
# Cloud output cleaning (from experiments/data_process/cloud_output_cleaner.py)
# =============================================================================

def _normalize_refusal(text: str) -> Optional[str]:
    """Check if text is a refusal and normalize it."""
    lowered = text.lower()
    if any(pattern in lowered for pattern in REFUSAL_PATTERNS):
        return "Sorry, I cannot answer."
    return None


def _strip_trailing_explanation(candidate: str) -> str:
    """Remove trailing explanations from SQL."""
    trimmed = candidate
    
    # Handle llama-3-sqlcoder issue: remove standalone "assistant" word and everything after
    # The pattern matches "assistant" (optionally followed by colon) as a standalone word,
    # not part of another identifier (e.g., "teaching_assistant")
    # This handles cases like: "SELECT * FROM t;assistant I am looking for..."
    assistant_pattern = r'(?<![a-zA-Z_])assistant:?\s*(?=[A-Z]|I[\'\s]|[A-Z][a-z]|from |to |$)'
    match = re.search(assistant_pattern, trimmed, re.IGNORECASE)
    if match:
        trimmed = trimmed[:match.start()].strip()
    
    # Also handle case where "assistant" appears right after semicolon
    # e.g., "SELECT * FROM t;assistant..." -> "SELECT * FROM t;"
    semicolon_assistant = re.search(r';(\s*)assistant\b', trimmed, re.IGNORECASE)
    if semicolon_assistant:
        trimmed = trimmed[:semicolon_assistant.start() + 1].strip()
    
    for marker in EXPLANATION_MARKERS:
        idx = trimmed.lower().find(marker)
        if idx != -1:
            trimmed = trimmed[:idx].strip()
    return trimmed.strip()


def _extract_sql_segment(cleaned: str) -> Optional[str]:
    """Extract SQL segment from cleaned text."""
    if not cleaned:
        return None

    keyword_regex = re.compile(r"(" + "|".join(SQL_KEYWORDS) + r")", re.IGNORECASE)
    match = keyword_regex.search(cleaned)
    if not match:
        return None

    sql_candidate = cleaned[match.start():].strip()
    if not sql_candidate:
        return None

    sql_candidate = _strip_trailing_explanation(sql_candidate)

    # If there is explanatory text after a terminating semicolon, drop it.
    if ";" in sql_candidate:
        last_semicolon = sql_candidate.rfind(";")
        tail = sql_candidate[last_semicolon + 1:].strip()
        if tail and not keyword_regex.search(tail):
            sql_candidate = sql_candidate[:last_semicolon + 1].strip()

    return sql_candidate or None


def _extract_from_incomplete_block(raw: str) -> Optional[str]:
    """Extract SQL from incomplete code block."""
    pattern = re.compile(r"```(?:[a-zA-Z0-9_]+)?\s*(.+)$", re.IGNORECASE | re.DOTALL)
    match = pattern.search(raw)
    if not match:
        return None

    snippet = match.group(1).strip().split("```", 1)[0].strip()
    if not snippet:
        return None

    cleaned = clean_model_response(snippet) or snippet
    return _extract_sql_segment(cleaned) or snippet


def clean_sql_from_cloud_output(sql_line: str) -> Optional[str]:
    """
    Clean and extract SQL from cloud inference output.
    (Aligned with experiments/data_process/cloud_output_cleaner.py)
    
    Handles various formats:
    1. Plain SQL: SELECT * FROM table;
    2. Code blocks: ```sql SELECT * FROM table; ```
    3. Inline code: `SELECT * FROM table;`
    4. Sorry responses: Sorry, I cannot answer.
    5. Error messages: Error: API timeout
    
    Args:
        sql_line: Raw line from cloud inference output
    
    Returns:
        Cleaned SQL string or None if not a valid SQL
    """
    if not sql_line or not sql_line.strip():
        return "Sorry, I cannot answer."

    sql_line = sql_line.strip()

    refusal = _normalize_refusal(sql_line)
    if refusal:
        return refusal

    if sql_line.startswith("Error:"):
        return sql_line

    if len(sql_line) < 3:
        return None

    cleaned = clean_model_response(sql_line)
    if not cleaned:
        cleaned = sql_line

    refusal = _normalize_refusal(cleaned)
    if refusal:
        return refusal

    extracted = _extract_sql_segment(cleaned)
    if extracted:
        return extracted

    fallback = _extract_from_incomplete_block(sql_line)
    if fallback:
        return fallback

    # Don't print invalid line warnings in batch mode
    return None


# =============================================================================
# High-level API for RBAC evaluation
# =============================================================================

def clean_sql_response(response: str) -> str:
    """
    Extract SQL from model response (simplified API).
    
    Args:
        response: Raw model response
        
    Returns:
        Cleaned SQL string
    """
    if not response:
        return ""
    
    cleaned = clean_sql_from_cloud_output(response)
    if cleaned is None:
        return response.strip()
    return cleaned


def is_refusal(response: str) -> bool:
    """
    Detect if response is a refusal to answer.
    
    Args:
        response: Model response
        
    Returns:
        True if response is detected as a refusal
    """
    if not response:
        return True
    
    return _normalize_refusal(response) is not None


def extract_sql_and_check_refusal(response: str) -> Tuple[str, bool]:
    """
    Extract SQL and check if response is a refusal.
    
    Args:
        response: Model response
        
    Returns:
        Tuple of (cleaned_sql, is_refusal)
    """
    if not response:
        return "", True
    
    # First check if it's a refusal
    if is_refusal(response):
        return "", True
    
    # Extract SQL
    sql = clean_sql_response(response)
    
    return sql, False


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison.
    
    Args:
        sql: SQL string
        
    Returns:
        Normalized SQL string
    """
    if not sql:
        return ""
    
    # Convert to lowercase
    sql = sql.lower()
    
    # Remove extra whitespace
    sql = " ".join(sql.split())
    
    # Remove trailing semicolon
    sql = sql.rstrip(";")
    
    return sql.strip()


def process_cloud_predictions(input_file: str, output_file: str) -> dict:
    """
    Process cloud inference predictions file and clean the SQL statements.
    (Aligned with experiments/data_process/cloud_output_cleaner.py)
    
    Args:
        input_file: Path to raw cloud inference output file  
        output_file: Path to cleaned output file for evaluation

    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total_lines': 0,
        'valid_sql': 0,
        'sorry_responses': 0,
        'error_responses': 0,
        'invalid_lines': 0,
        'code_blocks_cleaned': 0
    }

    # Read raw predictions
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    
    # Process each prediction
    for line_num, line in enumerate(lines, 1):
        stats['total_lines'] += 1
        
        # Clean SQL
        cleaned = clean_sql_from_cloud_output(line)
        
        # IMPORTANT: Replace invalid lines with "Sorry, I cannot answer." to maintain line count
        if cleaned is None:
            stats['invalid_lines'] += 1
            cleaned = "Sorry, I cannot answer."
        
        # Count response types
        if "Sorry" in cleaned:
            stats['sorry_responses'] += 1
        elif cleaned.startswith("Error:"):
            stats['error_responses'] += 1
        else:
            stats['valid_sql'] += 1
            
        if "```" in line:
            stats['code_blocks_cleaned'] += 1
        
        cleaned_lines.append(cleaned)
    
    # Write cleaned predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        if cleaned_lines:
            f.write('\n'.join(cleaned_lines))
            f.write('\n')
    
    return stats
