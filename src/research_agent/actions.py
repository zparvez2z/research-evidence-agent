"""Structured actions a decision model can select."""

from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass
class ToolAction:
    """Request execution of one named tool with keyword arguments."""

    tool: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise ValueError("tool must be a non-blank string")
        if not isinstance(self.arguments, dict):
            raise ValueError("arguments must be a dictionary")


@dataclass
class FinalAction:
    """Propose a user-facing answer grounded in identified evidence."""

    answer: str
    evidence_ids: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer must be a non-blank string")
        if not isinstance(self.evidence_ids, list) or any(
            not isinstance(evidence_id, str) or not evidence_id.strip()
            for evidence_id in self.evidence_ids
        ):
            raise ValueError("evidence_ids must be a list of non-blank strings")
        self.evidence_ids = list(dict.fromkeys(self.evidence_ids))


Action: TypeAlias = ToolAction | FinalAction
