"""Language-model-as-a-judge utilities for role assignments."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .data import RoleAssignment, RoleAssignmentCollection, RoleDefinition
from .loader import load_role_assignments

__all__ = [
    "RoleAssignment",
    "RoleAssignmentCollection",
    "RoleDefinition",
    "RoleAssignmentJudge",
    "EvaluationResult",
    "EvaluationSummary",
    "JudgeDecision",
    "DatasetComparisonSpec",
    "BatchEvaluationResult",
    "load_role_assignments",
    # Quality metrics
    "RoleQualityEvaluator",
    "QualityReport",
    "DenyRateMetrics",
    "SemanticMetrics",
    "OverlapMetrics",
    "print_quality_report",
    # CRUD-level quality metrics
    "CrudRoleQualityEvaluator",
    "CrudQualityReport",
    "CrudDenyRateMetrics",
    "CrudCoverageMetrics",
    "print_crud_quality_report",
    # Visualization
    "RoleQualityVisualizer",
]


def __getattr__(name: str) -> Any:
    if name in {
        "RoleAssignmentJudge",
        "EvaluationResult",
        "EvaluationSummary",
        "JudgeDecision",
        "DatasetComparisonSpec",
        "BatchEvaluationResult",
    }:
        module = import_module(".judge", __name__)
        return getattr(module, name)
    
    if name in {
        "RoleQualityEvaluator",
        "QualityReport",
        "DenyRateMetrics",
        "SemanticMetrics",
        "OverlapMetrics",
        "print_quality_report",
        # CRUD-level
        "CrudRoleQualityEvaluator",
        "CrudQualityReport",
        "CrudDenyRateMetrics",
        "CrudCoverageMetrics",
        "print_crud_quality_report",
    }:
        module = import_module(".quality_metrics", __name__)
        return getattr(module, name)
    
    if name == "RoleQualityVisualizer":
        module = import_module(".quality_visualizer", __name__)
        return getattr(module, name)
    
    raise AttributeError(f"module {__name__} has no attribute {name}")
