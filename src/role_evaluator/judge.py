"""LLM-judge evaluation workflow for role assignments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is absent
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable

from src.llm_oracle import Oracle
from src.llm_oracle.config import models as model_config

from .data import RoleAssignment, RoleAssignmentCollection
from .loader import load_role_assignments
from .prompts import DEFAULT_SYSTEM_PROMPT, build_judge_prompt


@dataclass
class JudgeDecision:
    """Single LLM judgement comparing two role assignment options."""

    database: str
    winner: str
    rationale: str
    confidence: str
    raw_answer: str
    metadata: Dict[str, object] = field(default_factory=dict)

    def normalized_winner(self) -> str:
        candidate = self.winner.strip().lower()
        if candidate in {"a", "option a"}:
            return "A"
        if candidate in {"b", "option b"}:
            return "B"
        if candidate in {"tie", "draw"}:
            return "tie"
        return "unknown"

    def to_dict(self) -> Dict[str, object]:
        return {
            "database": self.database,
            "winner": self.normalized_winner(),
            "rationale": self.rationale,
            "confidence": self.confidence,
            "raw_answer": self.raw_answer,
            "metadata": dict(self.metadata),
        }


@dataclass
class EvaluationSummary:
    """Aggregate statistics across all LLM judgements."""

    wins_a: int
    wins_b: int
    ties: int
    unknown: int

    @property
    def comparisons(self) -> int:
        return self.wins_a + self.wins_b + self.ties + self.unknown

    @property
    def decisive_comparisons(self) -> int:
        return self.wins_a + self.wins_b

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.decisive_comparisons if self.decisive_comparisons else 0.0

    @property
    def win_rate_b(self) -> float:
        return self.wins_b / self.decisive_comparisons if self.decisive_comparisons else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "ties": self.ties,
            "unknown": self.unknown,
            "comparisons": self.comparisons,
            "decisive_comparisons": self.decisive_comparisons,
            "win_rate_a": self.win_rate_a,
            "win_rate_b": self.win_rate_b,
        }


@dataclass
class EvaluationResult:
    """Full output bundle returned by the judge."""

    decisions: List[JudgeDecision]
    summary: EvaluationSummary
    assignment_a_path: str
    assignment_b_path: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "summary": self.summary.to_dict(),
            "assignment_a_path": self.assignment_a_path,
            "assignment_b_path": self.assignment_b_path,
        }


@dataclass
class DatasetComparisonSpec:
    """Configuration for evaluating a pair of role-assignment artifacts."""

    label: str
    assignment_a_path: str
    assignment_b_path: str
    databases: Optional[Sequence[str]] = None
    criteria: Optional[Iterable[str]] = None
    limit: Optional[int] = None
    show_progress: Optional[bool] = None


@dataclass
class BatchEvaluationResult:
    """Aggregated output across multiple dataset comparisons."""

    decisions: List[JudgeDecision]
    summary: EvaluationSummary
    results: Dict[str, EvaluationResult]

    def to_dict(self) -> Dict[str, object]:
        return {
            "summary": self.summary.to_dict(),
            "results": {label: result.to_dict() for label, result in self.results.items()},
            "decisions": [
                {**decision.to_dict(), "dataset": decision.metadata.get("dataset")}
                for decision in self.decisions
            ],
        }


class RoleAssignmentJudge:
    """Judge two role assignment artifacts using an LLM."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_completion_tokens: int = 800,
        azure_endpoint: str = "",
        max_parse_retries: int = 3,
    ) -> None:
        self.oracle = Oracle(model=model, apikey=api_key, azure_end_point=azure_endpoint)
        self.system_prompt = system_prompt
        if model in {getattr(model_config, "MODEL_GPT_5", None), getattr(model_config, "MODEL_GPT_5_MINI", None)}:
            self.temperature = 1.0
        else:
            self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.max_parse_retries = max(0, int(max_parse_retries))

    def judge_database(
        self,
        assignment_a: RoleAssignment,
        assignment_b: RoleAssignment,
        *,
        criteria: Optional[Iterable[str]] = None,
    ) -> JudgeDecision:
        user_prompt = build_judge_prompt(
            database=assignment_a.database,
            assignment_a=assignment_a,
            assignment_b=assignment_b,
            evaluation_criteria=criteria,
        )
        attempts_used = 0
        response: Dict[str, object] = {}
        parsed: Dict[str, str] = {}
        answer_text = ""
        success = False
        failure_reason = ""

        total_attempts = self.max_parse_retries + 1
        for attempt in range(total_attempts):
            attempts_used = attempt + 1
            response = self.oracle.query(
                prompt_sys=self.system_prompt,
                prompt_user=user_prompt,
                temp=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=self.max_completion_tokens,
            )
            raw_answer = response.get("answer", "")
            answer_text = str(raw_answer).strip()
            parsed = self._parse_answer(answer_text)

            if self._is_parse_success(answer_text, parsed):
                success = True
                break

            failure_reason = self._classify_retry_failure(answer_text, parsed)
            if attempt + 1 >= total_attempts:
                break

        metadata: Dict[str, object] = {
            "query": response.get("query"),
            "parsed": parsed,
            "attempts": attempts_used,
        }
        if attempts_used > 1:
            metadata["retry_success"] = success
        if not success and failure_reason:
            metadata["retry_reason"] = failure_reason

        return JudgeDecision(
            database=assignment_a.database,
            winner=parsed.get("winner", "unknown"),
            rationale=parsed.get("rationale", answer_text),
            confidence=parsed.get("confidence", "low"),
            raw_answer=answer_text,
            metadata=metadata,
        )

    def evaluate(
        self,
        assignment_a_path: str,
        assignment_b_path: str,
        *,
        databases: Optional[Sequence[str]] = None,
        criteria: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        show_progress: bool = True,
        progress_description: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> EvaluationResult:
        collection_a = load_role_assignments(assignment_a_path)
        collection_b = load_role_assignments(assignment_b_path)

        shared_databases = self._select_databases(collection_a, collection_b, databases, limit)
        if not shared_databases:
            raise ValueError("No overlapping databases found between the two role assignment files.")

        return self._evaluate_collections(
            collection_a,
            collection_b,
            shared_databases,
            criteria=criteria,
            show_progress=show_progress,
            progress_description=progress_description,
            progress_callback=progress_callback,
            assignment_a_path=str(assignment_a_path),
            assignment_b_path=str(assignment_b_path),
        )

    def evaluate_many(
        self,
        comparisons: Sequence[DatasetComparisonSpec],
        *,
        combined_progress: bool = True,
        dataset_progress: bool = False,
        progress_description: str = "Evaluating databases",
    ) -> BatchEvaluationResult:
        if not comparisons:
            raise ValueError("No comparison specifications provided.")

        planned: List[tuple[DatasetComparisonSpec, RoleAssignmentCollection, RoleAssignmentCollection, List[str]]] = []
        total_databases = 0

        for spec in comparisons:
            collection_a = load_role_assignments(spec.assignment_a_path)
            collection_b = load_role_assignments(spec.assignment_b_path)
            shared_databases = self._select_databases(collection_a, collection_b, spec.databases, spec.limit)
            if not shared_databases:
                raise ValueError(
                    f"No overlapping databases found for comparison '{spec.label}'."
                )
            planned.append((spec, collection_a, collection_b, shared_databases))
            total_databases += len(shared_databases)

        combined_bar = None
        if combined_progress and total_databases > 0:
            combined_bar = tqdm(total=total_databases, desc=progress_description, unit="db")

        results: Dict[str, EvaluationResult] = {}
        combined_decisions: List[JudgeDecision] = []

        for spec, collection_a, collection_b, shared_databases in planned:
            if spec.label in results:
                raise ValueError(f"Duplicate comparison label detected: {spec.label}")

            def make_callback() -> Optional[Callable[[str], None]]:
                if combined_bar is None:
                    return None

                def _callback(_: str) -> None:
                    combined_bar.update(1)

                return _callback

            per_result = self._evaluate_collections(
                collection_a,
                collection_b,
                shared_databases,
                criteria=spec.criteria,
                show_progress=spec.show_progress if spec.show_progress is not None else dataset_progress,
                progress_description=f"{spec.label}: databases" if dataset_progress or spec.show_progress else None,
                progress_callback=make_callback(),
                assignment_a_path=str(spec.assignment_a_path),
                assignment_b_path=str(spec.assignment_b_path),
            )

            for decision in per_result.decisions:
                decision.metadata.setdefault("dataset", spec.label)

            results[spec.label] = per_result
            combined_decisions.extend(per_result.decisions)

        if combined_bar is not None:
            combined_bar.close()

        summary = self._summarize(combined_decisions)
        return BatchEvaluationResult(decisions=combined_decisions, summary=summary, results=results)

    def _evaluate_collections(
        self,
        collection_a: RoleAssignmentCollection,
        collection_b: RoleAssignmentCollection,
        shared_databases: Sequence[str],
        *,
        criteria: Optional[Iterable[str]] = None,
        show_progress: bool = True,
        progress_description: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        assignment_a_path: str,
        assignment_b_path: str,
    ) -> EvaluationResult:
        iterator: Iterable[str]
        if show_progress and len(shared_databases) > 1:
            iterator = tqdm(shared_databases, desc=progress_description or "Judging databases", unit="db")
        else:
            iterator = shared_databases
        decisions: List[JudgeDecision] = []
        for database in iterator:
            assignment_a = collection_a.get(database)
            assignment_b = collection_b.get(database)
            if not assignment_a or not assignment_b:
                continue
            decision = self.judge_database(assignment_a, assignment_b, criteria=criteria)
            decisions.append(decision)
            if progress_callback:
                progress_callback(database)

        summary = self._summarize(decisions)
        return EvaluationResult(
            decisions=decisions,
            summary=summary,
            assignment_a_path=assignment_a_path,
            assignment_b_path=assignment_b_path,
        )

    def _select_databases(
        self,
        collection_a: RoleAssignmentCollection,
        collection_b: RoleAssignmentCollection,
        explicit: Optional[Sequence[str]],
        limit: Optional[int],
    ) -> List[str]:
        if explicit:
            selected = [db for db in explicit if collection_a.get(db) and collection_b.get(db)]
        else:
            shared = set(collection_a.databases()) & set(collection_b.databases())
            selected = sorted(shared)
        if limit is not None:
            selected = selected[:limit]
        return selected

    def _summarize(self, decisions: Sequence[JudgeDecision]) -> EvaluationSummary:
        wins_a = wins_b = ties = unknown = 0
        for decision in decisions:
            winner = decision.normalized_winner()
            if winner == "A":
                wins_a += 1
            elif winner == "B":
                wins_b += 1
            elif winner == "tie":
                ties += 1
            else:
                unknown += 1
        return EvaluationSummary(wins_a=wins_a, wins_b=wins_b, ties=ties, unknown=unknown)

    def _is_parse_success(self, answer_text: str, parsed: Dict[str, str]) -> bool:
        if not answer_text.strip():
            return False
        normalized = self._normalize_winner_token(parsed.get("winner"))
        return normalized in {"A", "B", "tie"}

    @staticmethod
    def _normalize_winner_token(value: Optional[str]) -> str:
        if value is None:
            return "unknown"
        candidate = str(value).strip().lower()
        if candidate in {"a", "option a"}:
            return "A"
        if candidate in {"b", "option b"}:
            return "B"
        if candidate in {"tie", "draw"}:
            return "tie"
        return "unknown"

    def _classify_retry_failure(self, answer_text: str, parsed: Dict[str, str]) -> str:
        if not answer_text.strip():
            return "empty_answer"
        normalized = self._normalize_winner_token(parsed.get("winner"))
        if normalized == "unknown":
            return "winner_unknown"
        return "unclassified"

    def _parse_answer(self, answer: str) -> Dict[str, str]:
        if not answer:
            return {}
        cleaned = answer.strip()
        pattern = re.compile(
            r"Winner\s*:\s*(?P<winner>[ABab]|tie)\b.*?Confidence\s*:\s*(?P<confidence>low|medium|high)\b.*?Rationale\s*:\s*(?P<rationale>.+)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(cleaned)
        if match:
            data = match.groupdict()
            winner = data.get("winner", "").strip().lower()
            if winner in {"a", "option a"}:
                mapped_winner = "A"
            elif winner in {"b", "option b"}:
                mapped_winner = "B"
            elif winner == "tie":
                mapped_winner = "tie"
            else:
                mapped_winner = "unknown"
            return {
                "winner": mapped_winner,
                "confidence": data.get("confidence", "low").strip().lower(),
                "rationale": data.get("rationale", "").strip(),
            }
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            extracted = self._extract_json_like(cleaned)
            if extracted:
                return extracted

        lowered = cleaned.lower()
        if "tie" in lowered:
            winner = "tie"
        elif "option b" in lowered or "b wins" in lowered or "choose b" in lowered:
            winner = "B"
        elif "option a" in lowered or "a wins" in lowered or "choose a" in lowered:
            winner = "A"
        else:
            winner = "unknown"
        return {"winner": winner, "rationale": cleaned, "confidence": "low"}

    def _extract_json_like(self, text: str) -> Dict[str, str] | None:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return {k: str(v) for k, v in parsed.items()}
        return None


__all__ = [
    "RoleAssignmentJudge",
    "JudgeDecision",
    "EvaluationSummary",
    "EvaluationResult",
    "DatasetComparisonSpec",
    "BatchEvaluationResult",
]
