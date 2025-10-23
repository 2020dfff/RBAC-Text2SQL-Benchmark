"""Utility helpers for normalizing model-generated SQL text."""

from __future__ import annotations

import re
from typing import Match

__all__ = [
    "clean_model_response",
    "strip_code_fences",
    "extract_last_sql_code_block",
]


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
    """Return the last fenced code block whose language is in ``allowed_langs``.

    If ``allowed_langs`` is ``None`` the language filter is skipped."""

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
    """Return the final fenced SQL code block within ``text`` if present."""

    return _extract_last_code_block(text, allowed_langs=SQL_LANG_ALIASES)


def strip_code_fences(text: str) -> str:
    """Remove fenced code blocks while preserving the enclosed SQL."""

    def _replace(match: Match[str]) -> str:
        inner = match.group("code") or ""
        return inner.strip()

    return CODE_BLOCK_RE.sub(_replace, text)


def clean_model_response(response: str, *, mode: str = "default") -> str:
    """Normalize raw LLM output to a compact SQL statement.

    Args:
        response: Raw model output text.
        mode: Cleaning mode. ``"default"`` preserves legacy behaviour, while
            ``"snowflake"`` enables heuristics tailored for Snowflake outputs.
    """

    if not response:
        return ""

    text = response.strip()
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

    return cleaned.strip()
