"""Utilities for loading role assignment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .data import RoleAssignmentCollection, ensure_path


def load_role_assignments(path: Path | str) -> RoleAssignmentCollection:
    """Load role assignments from a JSON file."""
    path = ensure_path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload: Mapping[str, Any] = json.load(handle)
    return RoleAssignmentCollection.from_raw(payload)


__all__ = ["load_role_assignments", "RoleAssignmentCollection"]
