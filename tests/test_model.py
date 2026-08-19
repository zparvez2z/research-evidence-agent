import pytest

from research_agent.actions import FinalAction, ToolAction
from research_agent.model import ScriptedModel


def test_scripted_model_returns_actions_in_order() -> None:
    first = ToolAction("calculate", {"expression": "2 + 2"})
    second = FinalAction("Four", [])
    model = ScriptedModel([first, second])

    assert model.decide("question", [], []) is first
    assert model.decide("question", [], []) is second


def test_scripted_model_raises_when_exhausted() -> None:
    model = ScriptedModel([])
    with pytest.raises(RuntimeError, match="no actions remaining"):
        model.decide("question", [], [])
