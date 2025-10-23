"""Utilities for cleaning and processing SQL outputs from cloud APIs."""

from __future__ import annotations

import re
from typing import Optional

from ..predict.response_cleaner import clean_model_response

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
    "unable to answer",
    "empty response",
    "not enough information",
    "no information",
    "the provided schema",
    "there is no column",
    "there is no information",
    "it seems that the information",
    "cannot be determined",
)


def _normalize_refusal(text: str) -> Optional[str]:
    lowered = text.lower()
    if any(pattern in lowered for pattern in REFUSAL_PATTERNS):
        return "Sorry, I cannot answer."
    return None


def _strip_trailing_explanation(candidate: str) -> str:
    trimmed = candidate
    
    # Handle llama-3-sqlcoder issue: remove standalone "assistant" word and everything after it
    # Pattern examples from actual outputs:
    # 1. SELECT ... FROM ...;assistant I am a bot...
    # 2. SELECT ... FROM ...;assistant: I'd be happy...  (with colon)
    # 3. SELECT ... FROM ... assistant I am a bot...
    # 4. ... WHERE Age > 20assistant I'm trying to learn...  (digit+assistant)
    # 5. ... GROUP BY Countryassistant I am here...  (word+assistant, but NO underscore)
    # 
    # We DON'T want to remove "assistant" when it's:
    # - Part of an identifier with underscore (assistant_id, sales_assistant)
    # - Inside a string literal ('assistant', "assistant")
    
    # The key insight: llama's "assistant" is ALWAYS followed by conversational text
    # starting with pronouns (I, I'm, I'd, I've) or question words (The, This, Can, What, How)
    
    # Match "assistant" when:
    # 1. NOT preceded by underscore (negative lookbehind for _)
    # 2. Followed by optional colon + whitespace + conversational markers
    # This catches: ";assistant I", "assistant: I'm", " assistant I", "20assistant I", "Countryassistant I"
    # But NOT: "sales_assistant", "assistant_id"
    pattern = r'(?<!_)assistant:?\s+(?=I[\'\s]|The |This |Can |What |How |Why |Sorry |Please )'
    match = re.search(pattern, trimmed, re.IGNORECASE)
    if match:
        # Only keep the SQL before the conversational "assistant"
        trimmed = trimmed[:match.start()].strip()
    
    for marker in EXPLANATION_MARKERS:
        idx = trimmed.lower().find(marker)
        if idx != -1:
            trimmed = trimmed[:idx].strip()
    return trimmed.strip()


def _extract_sql_segment(cleaned: str) -> Optional[str]:
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
        tail = sql_candidate[last_semicolon + 1 :].strip()
        if tail and not keyword_regex.search(tail):
            sql_candidate = sql_candidate[: last_semicolon + 1].strip()

    return sql_candidate or None


def _extract_from_incomplete_block(raw: str) -> Optional[str]:
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
    
    Handles various formats:
    1. Plain SQL: SELECT * FROM table;
    2. Code blocks: ```sql SELECT * FROM table; ```
    3. Inline code: `SELECT * FROM table;`
    4. Sorry responses: Sorry, I cannot answer this question.
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

    if len(sql_line) > 100:
        print(f"Invalid line detected: {sql_line[:100]}...")
    else:
        print(f"Invalid line detected: {sql_line}")
    return None

def process_cloud_predictions(input_file: str, output_file: str) -> dict:
    """
    Process cloud inference predictions file and clean the SQL statements.
    
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
        # This ensures the output file has the same number of lines as input for proper evaluation alignment
        if cleaned is None:
            stats['invalid_lines'] += 1
            cleaned = "Sorry, I cannot answer."
        
        # Count response types (invalid lines already converted to "Sorry" above)
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