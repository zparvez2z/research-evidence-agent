import pytest

from research_agent.tools import check_constraints


def test_check_constraints_passes_candidate() -> None:
    result = check_constraints(
        "distilled-hybrid",
        [{"field": "f1", "op": ">=", "value": 0.72}, {"field": "latency_ms", "op": "<", "value": 150}],
    )
    assert result["passed"] is True
    assert result["violations"] == []


def test_check_constraints_reports_failing_candidate() -> None:
    result = check_constraints(
        "large-api", [{"field": "latency_ms", "op": "<=", "value": 200}]
    )
    assert result["passed"] is False
    assert result["violations"] == result["checks"]
    assert result["checks"][0]["actual"] == 820


def test_check_constraints_handles_boolean_constraint() -> None:
    result = check_constraints(
        "quantized-small", [{"field": "local_inference", "op": "==", "value": True}]
    )
    assert result["passed"] is True


def test_check_constraints_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="Unknown metadata field"):
        check_constraints("lora-small", [{"field": "cost", "op": "<", "value": 10}])


def test_check_constraints_rejects_unsupported_operator() -> None:
    with pytest.raises(ValueError, match="Unsupported operator"):
        check_constraints("lora-small", [{"field": "f1", "op": "~=", "value": 0.74}])


def test_check_constraints_rejects_unknown_document() -> None:
    with pytest.raises(ValueError, match="Unknown document_id"):
        check_constraints("missing", [{"field": "f1", "op": ">", "value": 0.5}])
