"""Small offline demonstration of the explicit agent runtime."""

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent.actions import FinalAction, ToolAction
from research_agent.agent import ResearchAgent
from research_agent.model import ScriptedModel
from research_agent.tools.registry import create_default_registry


def main() -> None:
    """Run a fixed action sequence without optional model dependencies."""
    print("Deterministic ScriptedModel demonstration")
    actions = [
        ToolAction("search_notes", {"query": "LoRA Small F1"}),
        ToolAction("read_note", {"document_id": "lora-small"}),
        FinalAction("LoRA Small achieved an F1 score of 0.74.", ["lora-small"]),
    ]
    result = ResearchAgent(
        ScriptedModel(actions), create_default_registry()
    ).run("What F1 score did LoRA Small achieve?")

    for observation in result.state.observations:
        outcome = observation.error if observation.error else observation.result
        print(f"Step {observation.step}: {observation.tool} {observation.arguments}")
        print(f"  Observation: {outcome}")
    print(f"Final status: {result.status}")
    print(f"Final answer: {result.answer}")


if __name__ == "__main__":
    main()
