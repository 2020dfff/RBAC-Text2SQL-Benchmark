"""Prompt helpers for LLM judge evaluation."""

from __future__ import annotations

from typing import Iterable, List

from .data import RoleAssignment, RoleDefinition


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert database security consultant. Analyse role-based access proposals "
    "carefully and explain which option is more coherent and realistic for production use."
)


def _render_role(role: RoleDefinition) -> str:
    tables = ", ".join(role.tables) if role.tables else "<none>"
    description = role.description or "<no description>"
    return f"- {role.role or '<unnamed>'}: tables=({tables}) | {description}"


def _render_assignment(assignment: RoleAssignment) -> str:
    lines: List[str] = [f"Database `{assignment.database}`"]
    for entry in assignment.roles:
        lines.append(_render_role(entry))
    if len(assignment.roles) == 0:
        lines.append("- <no roles provided>")
    return "\n".join(lines)


def build_judge_prompt(
    database: str,
    assignment_a: RoleAssignment,
    assignment_b: RoleAssignment,
    evaluation_criteria: Iterable[str] | None = None,
) -> str:
    """Compose the user prompt for the judge model."""
    criteria_lines = list(evaluation_criteria or [
        "Coherence of the entire role portfolio for the target database. For example, from a possible unified scenario.",
        "Coverage of real-world responsibilities without redundant overlap.",
        "Clarity of role descriptions relative to the referenced tables.",
        "Adherence to best practices like least privilege and role interrelation.",
    ])

    criteria_text = "\n".join(f"- {criterion}" for criterion in criteria_lines)

    prompt = f"""
Compare two RBAC role proposals for the same database and decide which option is more coherent and realistic.

Database: {database}

Option A (roles listed in bullet form):
{_render_assignment(assignment_a)}

Option B (roles listed in bullet form):
{_render_assignment(assignment_b)}

Evaluation criteria:
{criteria_text}

Respond using exactly these three lines:
Winner: A | B | tie
Confidence: low | medium | high
Rationale: <concise explanation referencing the decisive factors>
"""
    return prompt.strip()


__all__ = ["DEFAULT_SYSTEM_PROMPT", "build_judge_prompt"]
