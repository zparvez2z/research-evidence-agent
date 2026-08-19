import pytest

from research_agent.state import AgentState


def test_agent_state_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="non-blank"):
        AgentState(" ")


def test_agent_state_records_success() -> None:
    state = AgentState("What is two plus two?")
    state.record_success(1, "calculate", {"expression": "2 + 2"}, 4)
    observation = state.observations[0]
    assert observation.result == 4
    assert observation.error is None


def test_agent_state_records_error() -> None:
    state = AgentState("Read a missing note")
    state.record_error(1, "read_note", {"document_id": "missing"}, "not found")
    observation = state.observations[0]
    assert observation.result is None
    assert observation.error == "not found"


def test_agent_state_derives_step_count_from_observations() -> None:
    state = AgentState("Calculate two values")
    assert state.step_count == 0
    state.record_success(1, "calculate", {"expression": "1 + 1"}, 2)
    state.record_success(2, "calculate", {"expression": "2 + 2"}, 4)
    assert state.step_count == 2
