import builtins

import pytest

from research_agent.actions import FinalAction, ToolAction
from research_agent.state import ToolObservation
from research_agent.tools.registry import ToolSpec
from research_agent.transformers_model import (
    SYSTEM_INSTRUCTION,
    TransformersDecisionModel,
    build_user_context,
    parse_model_action,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '{"type":"tool","tool":"search_notes","arguments":{"query":"LoRA"}}',
            ToolAction("search_notes", {"query": "LoRA"}),
        ),
        (
            '{"type":"final","answer":"F1 was 0.74.","evidence_ids":["lora-small"]}',
            FinalAction("F1 was 0.74.", ["lora-small"]),
        ),
        (
            '  {"type":"tool","tool":"calculate","arguments":{"expression":"2+2"}} \n',
            ToolAction("calculate", {"expression": "2+2"}),
        ),
        (
            '```json\n{"type":"final","answer":"Supported.","evidence_ids":["note"]}\n```',
            FinalAction("Supported.", ["note"]),
        ),
    ],
)
def test_parse_model_action_accepts_supported_json(text: str, expected: object) -> None:
    assert parse_model_action(text) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "Invalid model action JSON"),
        ('{"type":"other"}', "Unknown model action type"),
        ('{"type":"tool","arguments":{}}', "exactly"),
        ('{"type":"tool","tool":"read_note","arguments":[]}', "must be an object"),
        ('{"type":"final","evidence_ids":[]}', "exactly"),
        ('{"type":"final","answer":"No","evidence_ids":"note"}', "array of strings"),
    ],
)
def test_parse_model_action_rejects_invalid_responses(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_model_action(text)


def test_context_contains_only_explicit_observable_fields() -> None:
    tools = [ToolSpec("search_notes", "Search notes.", {"type": "object"})]
    observations = [
        ToolObservation(1, "search_notes", {"query": "LoRA"}, result=["lora-small"]),
        ToolObservation(2, "read_note", {"document_id": "missing"}, error="not found"),
    ]

    context = build_user_context("What was measured?", observations, tools)

    assert '"question": "What was measured?"' in context
    assert '"name": "search_notes"' in context
    assert '"result"' in context and '"lora-small"' in context
    assert '"error": "not found"' in context
    assert "chain-of-thought" not in context.lower()
    assert "rationale" not in context.lower()


def test_system_instruction_does_not_request_hidden_reasoning() -> None:
    assert "do not output analysis, rationale" in SYSTEM_INSTRUCTION.lower()
    assert "scratchpad" not in SYSTEM_INSTRUCTION.lower()


def test_missing_optional_dependencies_produce_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"torch", "transformers"}:
            raise ImportError(f"No module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    with pytest.raises(RuntimeError, match="optional Colab/model dependencies"):
        TransformersDecisionModel()
