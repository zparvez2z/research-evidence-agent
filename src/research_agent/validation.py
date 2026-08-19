"""Deterministic provenance checks for proposed final answers."""

from dataclasses import dataclass
from typing import Any

from .actions import FinalAction
from .state import AgentState


@dataclass
class ValidationResult:
    """Whether a final action meets the runtime's small provenance rules."""

    accepted: bool
    reasons: list[str]


def _successfully_read_ids(state: AgentState) -> set[str]:
    document_ids: set[str] = set()
    for observation in state.observations:
        if observation.tool != "read_note" or observation.error is not None:
            continue
        result: Any = observation.result
        if isinstance(result, dict) and isinstance(result.get("document_id"), str):
            document_ids.add(result["document_id"])
    return document_ids


def validate_final_action(
    state: AgentState, action: FinalAction
) -> ValidationResult:
    """Check that a final action cites documents successfully read this run."""
    read_ids = _successfully_read_ids(state)
    reasons: list[str] = []

    if not read_ids:
        reasons.append("No read_note call completed successfully.")
    if not action.evidence_ids:
        reasons.append("Final action must include at least one evidence ID.")

    unread_ids = [
        evidence_id for evidence_id in action.evidence_ids if evidence_id not in read_ids
    ]
    if unread_ids:
        reasons.append(
            "Evidence IDs were not successfully read: " + ", ".join(unread_ids) + "."
        )

    return ValidationResult(accepted=not reasons, reasons=reasons)
