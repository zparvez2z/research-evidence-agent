"""The explicit single-agent observe-decide-act runtime."""

from dataclasses import dataclass

from .actions import FinalAction, ToolAction
from .model import DecisionModel
from .state import AgentState
from .tools.registry import ToolRegistry
from .validation import validate_final_action


@dataclass
class AgentRunResult:
    """The externally meaningful outcome and inspectable state of one run."""

    answer: str | None
    evidence_ids: list[str]
    state: AgentState
    status: str


class ResearchAgent:
    """Ask a model for actions while Python executes and validates them safely."""

    def __init__(
        self, model: DecisionModel, registry: ToolRegistry, max_steps: int = 6
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.model = model
        self.registry = registry
        self.max_steps = max_steps

    def run(self, question: str) -> AgentRunResult:
        """Run at most ``max_steps`` model decisions for one question."""
        state = AgentState(question)

        for _ in range(self.max_steps):
            action = self.model.decide(
                state.question, state.observations, self.registry.list_specs()
            )
            if isinstance(action, FinalAction):
                validation = validate_final_action(state, action)
                if validation.accepted:
                    return AgentRunResult(
                        action.answer, list(action.evidence_ids), state, "completed"
                    )
                return AgentRunResult(None, list(action.evidence_ids), state, "final_rejected")

            if not isinstance(action, ToolAction):
                raise TypeError(f"Unsupported action type: {type(action).__name__}")

            step = state.step_count + 1
            try:
                result = self.registry.execute(action.tool, action.arguments)
            except Exception as error:
                state.record_error(
                    step,
                    action.tool,
                    action.arguments,
                    f"{type(error).__name__}: {error}",
                )
            else:
                state.record_success(step, action.tool, action.arguments, result)

        return AgentRunResult(None, [], state, "max_steps")
