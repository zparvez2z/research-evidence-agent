from eval.evaluate import check_expectations
from research_agent.agent import AgentRunResult
from research_agent.state import AgentState


def _result(*, status: str = "completed", answer: str | None = "F1 is 0.74") -> AgentRunResult:
    state = AgentState("Question")
    state.record_success(
        1,
        "read_note",
        {"document_id": "lora-small"},
        {"document_id": "lora-small"},
    )
    return AgentRunResult(answer, ["lora-small"], state, status)


def test_matching_result_passes() -> None:
    failures = check_expectations(
        _result(),
        {
            "status": "completed",
            "required_evidence_ids": ["lora-small"],
            "required_tools": ["read_note"],
            "answer_contains": ["0.74"],
        },
    )
    assert failures == []


def test_missing_required_evidence_fails() -> None:
    assert check_expectations(
        _result(), {"required_evidence_ids": ["large-api"]}
    ) == ["missing required evidence ID 'large-api'"]


def test_missing_required_tool_fails() -> None:
    assert check_expectations(_result(), {"required_tools": ["calculate"]}) == [
        "required tool 'calculate' was not used"
    ]


def test_wrong_status_fails() -> None:
    assert check_expectations(_result(status="max_steps"), {"status": "completed"})


def test_missing_answer_substring_fails() -> None:
    assert check_expectations(_result(), {"answer_contains": ["3 GB"]}) == [
        "answer did not contain '3 GB'"
    ]


def test_error_count_expectation_passes_and_fails() -> None:
    result = _result()
    result.state.record_error(2, "read_note", {"document_id": "missing"}, "missing")
    assert check_expectations(result, {"minimum_error_count": 1}) == []
    assert check_expectations(result, {"minimum_error_count": 2})


def test_maximum_tool_count_expectation_passes_and_fails() -> None:
    result = _result()
    assert check_expectations(result, {"maximum_tool_count": 1}) == []
    assert check_expectations(result, {"maximum_tool_count": 0})
