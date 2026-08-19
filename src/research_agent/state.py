"""Small, observable state structures for a future agent runtime."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolObservation:
    """The externally meaningful outcome of one tool call."""

    step: int
    tool: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("step must be an integer greater than or equal to 1")
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise ValueError("tool must be a non-blank string")
        if not isinstance(self.arguments, dict):
            raise ValueError("arguments must be a dictionary")
        if self.error is not None and (
            not isinstance(self.error, str) or not self.error.strip()
        ):
            raise ValueError("error must be a non-blank string when present")
        if self.result is not None and self.error is not None:
            raise ValueError("an observation cannot contain both a result and an error")


@dataclass
class AgentState:
    """The question and tool observations visible to the decision model."""

    question: str
    observations: list[ToolObservation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-blank string")

    def record_success(
        self, step: int, tool: str, arguments: dict[str, Any], result: Any
    ) -> None:
        self.observations.append(
            ToolObservation(step=step, tool=tool, arguments=arguments, result=result)
        )

    def record_error(
        self, step: int, tool: str, arguments: dict[str, Any], error: str
    ) -> None:
        self.observations.append(
            ToolObservation(step=step, tool=tool, arguments=arguments, error=error)
        )

    @property
    def step_count(self) -> int:
        """Return the number of recorded tool observations."""
        return len(self.observations)
