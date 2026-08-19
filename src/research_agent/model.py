"""Provider-neutral model interface and deterministic test implementation."""

from collections.abc import Sequence
from typing import Protocol

from .actions import Action
from .state import ToolObservation
from .tools.registry import ToolSpec


class DecisionModel(Protocol):
    """Something that selects the next action from observable context."""

    def decide(
        self,
        question: str,
        observations: Sequence[ToolObservation],
        tools: Sequence[ToolSpec],
    ) -> Action:
        """Select the next action."""
        ...


class ScriptedModel:
    """Return pre-created actions in order for deterministic tests."""

    def __init__(self, actions: Sequence[Action]) -> None:
        self._actions = list(actions)
        self._next_index = 0

    def decide(
        self,
        question: str,
        observations: Sequence[ToolObservation],
        tools: Sequence[ToolSpec],
    ) -> Action:
        if self._next_index >= len(self._actions):
            raise RuntimeError("ScriptedModel has no actions remaining")
        action = self._actions[self._next_index]
        self._next_index += 1
        return action
