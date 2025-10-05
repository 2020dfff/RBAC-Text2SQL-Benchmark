"""Data models for role assignment evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass
class RoleDefinition:
    """Structured representation of a single role assignment."""

    role: str
    description: str
    tables: List[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: Mapping[str, object]) -> "RoleDefinition":
        """Build a role definition from a raw dictionary."""
        role_name = str(raw.get("role", "")).strip()
        description = str(raw.get("description", "")).strip()
        tables_raw = raw.get("tables", [])

        if isinstance(tables_raw, str):
            tables = [segment.strip() for segment in tables_raw.split(",") if segment.strip()]
        elif isinstance(tables_raw, Sequence):
            tables = [str(segment).strip() for segment in tables_raw if str(segment).strip()]
        else:
            tables = []

        return cls(role=role_name, description=description, tables=tables)

    def to_dict(self) -> Dict[str, object]:
        """Serialize the role definition to a dictionary."""
        return {
            "role": self.role,
            "description": self.description,
            "tables": list(self.tables),
        }


@dataclass
class RoleAssignment:
    """Role assignments for a single database."""

    database: str
    roles: List[RoleDefinition]

    def to_dict(self) -> Dict[str, object]:
        """Serialize the assignment for storage or logging."""
        return {
            "database": self.database,
            "roles": [role.to_dict() for role in self.roles],
        }


@dataclass
class RoleAssignmentCollection:
    """Container that stores role assignments keyed by database name."""

    assignments: Dict[str, RoleAssignment]
    metadata: Dict[str, object] = field(default_factory=dict)

    def databases(self) -> List[str]:
        """Return a sorted list of database identifiers."""
        return sorted(self.assignments.keys())

    def get(self, database: str) -> Optional[RoleAssignment]:
        """Retrieve the assignment for a database if present."""
        return self.assignments.get(database)

    @classmethod
    def from_raw(cls, raw: Mapping[str, object]) -> "RoleAssignmentCollection":
        """Construct a collection from a raw JSON-like mapping."""
        assignments_raw = raw.get("assignments", {})
        metadata = raw.get("metadata", {})
        assignments: Dict[str, RoleAssignment] = {}

        if isinstance(assignments_raw, Mapping):
            iterator: Iterable = assignments_raw.items()
        elif isinstance(assignments_raw, Sequence):
            # Some files might store assignments as a list of dicts with keys embedded.
            iterator = ((item.get("database"), item.get("roles")) for item in assignments_raw if isinstance(item, Mapping))
        else:
            iterator = []

        for db_name, roles_raw in iterator:
            if not db_name:
                continue
            roles: List[RoleDefinition] = []
            if isinstance(roles_raw, Sequence):
                roles = [RoleDefinition.from_raw(entry) for entry in roles_raw if isinstance(entry, Mapping)]
            assignments[str(db_name)] = RoleAssignment(database=str(db_name), roles=roles)

        metadata_dict = metadata if isinstance(metadata, Mapping) else {}
        return cls(assignments=assignments, metadata=dict(metadata_dict))

    def to_dict(self) -> Dict[str, object]:
        """Serialize the full collection."""
        return {
            "assignments": {db: assignment.to_dict()["roles"] for db, assignment in self.assignments.items()},
            "metadata": dict(self.metadata),
        }


def ensure_path(path_like: Path | str) -> Path:
    """Convert path-like input to a Path object."""
    return path_like if isinstance(path_like, Path) else Path(path_like)
