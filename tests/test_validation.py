from research_agent.actions import FinalAction
from research_agent.state import AgentState
from research_agent.tools import read_note
from research_agent.validation import validate_final_action


def _state_with_reads(*document_ids: str) -> AgentState:
    state = AgentState("Question")
    for step, document_id in enumerate(document_ids, start=1):
        state.record_success(
            step, "read_note", {"document_id": document_id}, read_note(document_id)
        )
    return state


def test_validator_reports_no_reads_and_empty_evidence_in_stable_order() -> None:
    result = validate_final_action(AgentState("Question"), FinalAction("Answer", []))
    assert result.accepted is False
    assert result.reasons == [
        "No read_note call completed successfully.",
        "Final action must include at least one evidence ID.",
    ]


def test_validator_accepts_valid_evidence() -> None:
    result = validate_final_action(
        _state_with_reads("distilled-hybrid"),
        FinalAction("Answer", ["distilled-hybrid"]),
    )
    assert result.accepted is True
    assert result.reasons == []


def test_validator_rejects_unknown_or_unread_evidence() -> None:
    result = validate_final_action(
        _state_with_reads("distilled-hybrid"), FinalAction("Answer", ["large-api"])
    )
    assert result.reasons == ["Evidence IDs were not successfully read: large-api."]


def test_validator_accepts_multiple_read_evidence_ids() -> None:
    result = validate_final_action(
        _state_with_reads("distilled-hybrid", "large-api"),
        FinalAction("Answer", ["large-api", "distilled-hybrid"]),
    )
    assert result.accepted is True


def test_validator_ignores_failed_read_observation() -> None:
    state = AgentState("Question")
    state.record_error(1, "read_note", {"document_id": "missing"}, "missing")
    result = validate_final_action(state, FinalAction("Answer", ["missing"]))
    assert result.reasons == [
        "No read_note call completed successfully.",
        "Evidence IDs were not successfully read: missing.",
    ]
