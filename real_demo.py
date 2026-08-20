"""Terminal runner for the optional real-model Research Evidence Agent demo."""

from __future__ import annotations

import argparse
import json

import research_agent.transformers_model as transformers_model
from research_agent.agent import ResearchAgent
from research_agent.tools.registry import create_default_registry
from research_agent.transformers_model import TransformersDecisionModel


DEFAULT_QUESTIONS = [
    "What F1 score did LoRA Small achieve?",
    (
        "Which experiment meets F1 of at least 0.72, "
        "latency below 150 ms, and local inference?"
    ),
    "Which experiment used the least energy?",
]


def print_run(result: object) -> None:
    """Print only observable runtime state and the final result."""
    print("STATUS:", result.status)
    print("QUESTION:", result.state.question)
    for observation in result.state.observations:
        print(f"\nSTEP: {observation.step}")
        print("TOOL:", observation.tool)
        print("ARGUMENTS:", json.dumps(observation.arguments, sort_keys=True))
        if observation.error is not None:
            print("ERROR:", observation.error)
        else:
            print("RESULT:", json.dumps(observation.result, indent=2, sort_keys=True))
    print("\nFINAL ANSWER:", result.answer)
    print("EVIDENCE IDS:", result.evidence_ids)


def enable_action_json_debug() -> None:
    """Print the model's action object before strict parsing for diagnostics."""
    original_parse = transformers_model.parse_model_action

    def debug_parse(text: str):
        print("\n--- MODEL ACTION OUTPUT ---")
        try:
            value = json.loads(transformers_model._strip_json_fence(text))
            print(json.dumps(value, indent=2, sort_keys=True))
            if isinstance(value, dict):
                print("TOP-LEVEL KEYS:", sorted(value))
        except Exception:
            print(repr(text[:1000]))
        print("--- END MODEL ACTION OUTPUT ---\n")
        return original_parse(text)

    transformers_model.parse_model_action = debug_parse


def run_question(agent: ResearchAgent, question: str) -> None:
    """Run one real-model question and surface adapter/runtime errors clearly."""
    print("\n" + "=" * 72)
    try:
        result = agent.run(question)
    except (ValueError, RuntimeError) as error:
        print("REAL MODEL DEMO ERROR:", f"{type(error).__name__}: {error}")
        return
    print_run(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        action="append",
        help="Question to run. Repeat the flag for multiple questions. Defaults to three demo cases.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-2B",
        help="Hugging Face model name.",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--debug-action-json",
        action="store_true",
        help="Print the model's generated action JSON before strict parsing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug_action_json:
        enable_action_json_debug()

    print("Real open-weight model demonstration")
    print("MODEL:", args.model)
    model = TransformersDecisionModel(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )
    agent = ResearchAgent(
        model=model,
        registry=create_default_registry(),
        max_steps=args.max_steps,
    )

    questions = args.question or DEFAULT_QUESTIONS
    for question in questions:
        run_question(agent, question)


if __name__ == "__main__":
    main()
