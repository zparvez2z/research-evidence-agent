import pytest

from research_agent.actions import FinalAction, ToolAction
from research_agent.agent import ResearchAgent
from research_agent.model import ScriptedModel
from research_agent.tools.registry import create_default_registry


def _run(actions: list[ToolAction | FinalAction], max_steps: int = 6):
    return ResearchAgent(
        ScriptedModel(actions), create_default_registry(), max_steps=max_steps
    ).run("Which model fits?")


def test_successful_multi_step_run() -> None:
    result = _run(
        [
            ToolAction("search_notes", {"query": "distilled hybrid"}),
            ToolAction("read_note", {"document_id": "distilled-hybrid"}),
            FinalAction("Use the distilled hybrid.", ["distilled-hybrid"]),
        ]
    )
    assert result.status == "completed"
    assert result.answer == "Use the distilled hybrid."
    assert result.evidence_ids == ["distilled-hybrid"]
    assert len(result.state.observations) == 2
    assert [item.step for item in result.state.observations] == [1, 2]


def test_adaptive_path_executes_all_tools_before_finalization() -> None:
    result = _run(
        [
            ToolAction("search_notes", {"query": "distilled"}),
            ToolAction("read_note", {"document_id": "distilled-hybrid"}),
            ToolAction("calculate", {"expression": "73 / 100"}),
            FinalAction("Its F1 is 0.73.", ["distilled-hybrid"]),
        ]
    )
    assert result.status == "completed"
    assert [item.tool for item in result.state.observations] == [
        "search_notes", "read_note", "calculate"
    ]


def test_tool_error_recovery_and_error_format() -> None:
    result = _run(
        [
            ToolAction("read_note", {"document_id": "missing"}),
            ToolAction("search_notes", {"query": "distilled"}),
            ToolAction("read_note", {"document_id": "distilled-hybrid"}),
            FinalAction("Found it.", ["distilled-hybrid"]),
        ]
    )
    assert result.status == "completed"
    assert result.state.observations[0].error == (
        "ValueError: Unknown document_id: 'missing'"
    )
    assert len(result.state.observations) == 3


def test_unknown_tool_error_does_not_stop_run() -> None:
    result = _run(
        [
            ToolAction("not_a_tool", {}),
            ToolAction("read_note", {"document_id": "large-api"}),
            FinalAction("Read valid evidence.", ["large-api"]),
        ]
    )
    assert result.status == "completed"
    assert result.state.observations[0].error == "ValueError: Unknown tool: 'not_a_tool'"


def test_final_answer_is_rejected_before_reading_source() -> None:
    result = _run([FinalAction("Premature.", ["large-api"])])
    assert result.status == "final_rejected"
    assert result.answer is None
    assert result.state.observations == []


def test_final_answer_rejects_unread_evidence() -> None:
    result = _run(
        [
            ToolAction("read_note", {"document_id": "large-api"}),
            FinalAction("Wrong citation.", ["distilled-hybrid"]),
        ]
    )
    assert result.status == "final_rejected"
    assert result.answer is None


def test_max_steps_counts_decisions_and_does_not_execute_beyond_limit() -> None:
    result = _run(
        [
            ToolAction("calculate", {"expression": "1"}),
            ToolAction("calculate", {"expression": "2"}),
            ToolAction("calculate", {"expression": "3"}),
        ],
        max_steps=2,
    )
    assert result.status == "max_steps"
    assert result.answer is None
    assert [item.result for item in result.state.observations] == [1, 2]


@pytest.mark.parametrize("max_steps", [0, -1, 1.5, True])
def test_max_steps_must_be_a_positive_non_boolean_integer(max_steps: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ResearchAgent(
            ScriptedModel([]), create_default_registry(), max_steps=max_steps
        )  # type: ignore[arg-type]
