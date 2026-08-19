import pytest

from research_agent.tools.registry import create_default_registry


def test_registry_exposes_exactly_four_named_tools() -> None:
    names = [spec.name for spec in create_default_registry().list_specs()]
    assert names == ["search_notes", "read_note", "calculate", "check_constraints"]


def test_registry_executes_search_notes() -> None:
    result = create_default_registry().execute("search_notes", {"query": "quantized"})
    assert result[0]["document_id"] == "quantized-small"


def test_registry_executes_read_note() -> None:
    result = create_default_registry().execute(
        "read_note", {"document_id": "distilled-hybrid"}
    )
    assert result["document_id"] == "distilled-hybrid"


def test_registry_executes_calculate() -> None:
    assert create_default_registry().execute("calculate", {"expression": "6 * 7"}) == 42


def test_registry_executes_check_constraints() -> None:
    result = create_default_registry().execute(
        "check_constraints",
        {
            "document_id": "quantized-small",
            "requirements": [{"field": "local_inference", "op": "==", "value": True}],
        },
    )
    assert result["passed"] is True


def test_registry_rejects_unknown_tool() -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        create_default_registry().execute("missing", {})


def test_registry_rejects_non_dictionary_arguments() -> None:
    with pytest.raises(ValueError, match="dictionary"):
        create_default_registry().execute("calculate", [])  # type: ignore[arg-type]


def test_registry_propagates_underlying_tool_errors() -> None:
    with pytest.raises(ValueError, match="zero"):
        create_default_registry().execute("calculate", {"expression": "1 / 0"})
