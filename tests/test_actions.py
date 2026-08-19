import pytest

from research_agent.actions import FinalAction, ToolAction


def test_tool_action_valid_construction() -> None:
    action = ToolAction("calculate", {"expression": "2 + 2"})
    assert action.tool == "calculate"
    assert action.arguments == {"expression": "2 + 2"}


def test_tool_action_rejects_blank_tool() -> None:
    with pytest.raises(ValueError, match="non-blank"):
        ToolAction(" ", {})


def test_tool_action_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="dictionary"):
        ToolAction("calculate", [])  # type: ignore[arg-type]


def test_final_action_valid_construction() -> None:
    action = FinalAction("The answer is supported.", ["note-a"])
    assert action.evidence_ids == ["note-a"]


def test_final_action_rejects_blank_answer() -> None:
    with pytest.raises(ValueError, match="non-blank"):
        FinalAction(" ", [])


@pytest.mark.parametrize("evidence_ids", [[""], ["valid", "  "], [1]])
def test_final_action_rejects_invalid_evidence_ids(evidence_ids: object) -> None:
    with pytest.raises(ValueError, match="non-blank strings"):
        FinalAction("Answer", evidence_ids)  # type: ignore[arg-type]


def test_final_action_removes_duplicate_evidence_ids_in_order() -> None:
    action = FinalAction("Answer", ["note-b", "note-a", "note-b"])
    assert action.evidence_ids == ["note-b", "note-a"]
