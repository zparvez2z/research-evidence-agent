"""Deterministically check requirements against source note metadata."""

import operator
from collections.abc import Callable
from typing import Any

from .notes import read_note


_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def check_constraints(
    document_id: str, requirements: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check requirements against metadata loaded by document ID."""
    if not isinstance(requirements, list):
        raise ValueError("requirements must be a list")

    metadata = read_note(document_id)["metadata"]
    checks: list[dict[str, Any]] = []
    for index, requirement in enumerate(requirements):
        if not isinstance(requirement, dict):
            raise ValueError(f"Requirement {index} must be a dictionary")
        missing = {"field", "op", "value"} - requirement.keys()
        if missing:
            raise ValueError(f"Requirement {index} is missing: {', '.join(sorted(missing))}")

        field = requirement["field"]
        operation_name = requirement["op"]
        if not isinstance(field, str) or field not in metadata:
            raise ValueError(f"Unknown metadata field: {field!r}")
        if not isinstance(operation_name, str) or operation_name not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {operation_name!r}")

        actual = metadata[field]
        expected = requirement["value"]
        try:
            passed = _OPERATORS[operation_name](actual, expected)
        except TypeError as error:
            raise ValueError(
                f"Cannot compare field {field!r} value {actual!r} to {expected!r}"
            ) from error
        checks.append(
            {
                "field": field,
                "operator": operation_name,
                "expected": expected,
                "actual": actual,
                "passed": passed,
            }
        )

    violations = [check for check in checks if not check["passed"]]
    return {
        "document_id": document_id,
        "passed": not violations,
        "checks": checks,
        "violations": violations,
    }
