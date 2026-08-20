"""Deterministic evaluation of observable ResearchAgent runtime behavior."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_agent.agent import AgentRunResult, ResearchAgent
from research_agent.model import ScriptedModel
from research_agent.tools.registry import create_default_registry
from research_agent.transformers_model import parse_model_action


CASES_PATH = Path(__file__).with_name("cases.json")
EVALUATION_MAX_STEPS = 6


@dataclass
class CaseResult:
    """Concise deterministic outcome for one evaluation case."""

    case_id: str
    passed: bool
    status: str
    tool_count: int
    evidence_ids: list[str]
    failures: list[str]


def check_expectations(
    result: AgentRunResult, expectations: dict[str, Any]
) -> list[str]:
    """Return readable failures for the small supported expectation vocabulary."""
    failures: list[str] = []
    observations = result.state.observations
    used_tools = {observation.tool for observation in observations}

    expected_status = expectations.get("status")
    if expected_status is not None and result.status != expected_status:
        failures.append(f"status was {result.status!r}, expected {expected_status!r}")

    for evidence_id in expectations.get("required_evidence_ids", []):
        if evidence_id not in result.evidence_ids:
            failures.append(f"missing required evidence ID {evidence_id!r}")

    for tool in expectations.get("required_tools", []):
        if tool not in used_tools:
            failures.append(f"required tool {tool!r} was not used")

    answer = (result.answer or "").lower()
    for substring in expectations.get("answer_contains", []):
        if substring.lower() not in answer:
            failures.append(f"answer did not contain {substring!r}")

    error_count = sum(item.error is not None for item in observations)
    minimum_errors = expectations.get("minimum_error_count")
    if minimum_errors is not None and error_count < minimum_errors:
        failures.append(f"error count was {error_count}, expected at least {minimum_errors}")

    maximum_tools = expectations.get("maximum_tool_count")
    if maximum_tools is not None and len(observations) > maximum_tools:
        failures.append(
            f"tool count was {len(observations)}, expected at most {maximum_tools}"
        )

    return failures


def evaluate_case(case: dict[str, Any]) -> CaseResult:
    """Run one scripted case through the normal parser, model, tools, and agent."""
    actions = [
        parse_model_action(json.dumps(action)) for action in case["scripted_actions"]
    ]
    agent = ResearchAgent(
        ScriptedModel(actions),
        create_default_registry(),
        max_steps=EVALUATION_MAX_STEPS,
    )
    run_result = agent.run(case["question"])
    failures = check_expectations(run_result, case["expectations"])
    return CaseResult(
        case_id=case["id"],
        passed=not failures,
        status=run_result.status,
        tool_count=len(run_result.state.observations),
        evidence_ids=run_result.evidence_ids,
        failures=failures,
    )


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    """Load the transparent JSON case set."""
    with path.open(encoding="utf-8") as case_file:
        cases = json.load(case_file)
    if not isinstance(cases, list):
        raise ValueError("Evaluation cases must be a JSON array")
    return cases


def main() -> int:
    """Run all cases, print a concise report, and return a process exit code."""
    results = [evaluate_case(case) for case in load_cases()]
    print("Evaluation: Research Evidence Agent\n")
    for result in results:
        label = "PASS" if result.passed else "FAIL"
        print(f"{label} {result.case_id}")
        for failure in result.failures:
            print(f"  - {failure}")
    passed = sum(result.passed for result in results)
    print(f"\n{passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
