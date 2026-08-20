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
            '{"action":"search_notes","arguments":{"query":"LoRA"}}',
            ToolAction("search_notes", {"query": "LoRA"}),
        ),
        (
            '{"action":"final","answer":"F1 was 0.74.","evidence_ids":["lora-small"]}',
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
        ('{"action":"search_notes","arguments":[]}', "must be an object"),
        ('{"action":"final","answer":"No","evidence_ids":"note"}', "array of strings"),
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
    assert "thinking" not in SYSTEM_INSTRUCTION.lower()
    assert "reasoning" not in SYSTEM_INSTRUCTION.lower()


def test_system_instruction_directs_search_progress_and_avoids_repeats() -> None:
    instruction = SYSTEM_INSTRUCTION.lower()
    assert "after finding a relevant result, normally call read_note" in instruction
    assert "do not repeat the same successful search" in instruction
    assert "read a relevant document next" in instruction
    assert 'tool action: {"action":"<tool name>"' in instruction
    assert 'final action: {"action":"final"' in instruction


def test_default_model_is_qwen_3_5(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_names: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(name: str) -> str:
            return name

    class FakeLoadedModel:
        def to(self, device: object) -> None:
            assert device == "cpu"

        def eval(self) -> None:
            pass

    class FakeLoader:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: object) -> object:
            loaded_names.append(model_name)
            assert kwargs == {}
            return FakeLoadedModel() if len(loaded_names) == 2 else object()

    monkeypatch.setattr(
        "research_agent.transformers_model._import_model_dependencies",
        lambda: (FakeTorch(), FakeLoader, FakeLoader),
    )

    TransformersDecisionModel()

    assert loaded_names == ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B"]


def test_missing_optional_dependencies_produce_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"torch", "transformers"}:
            raise ImportError(f"No module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    with pytest.raises(RuntimeError, match="optional Colab/model dependencies"):
        TransformersDecisionModel()


def test_decide_supports_batch_encoding_outputs() -> None:
    class FakeInputIds:
        shape = (1, 3)

    class FakeBatchEncoding(dict):
        def to(self, device: object) -> "FakeBatchEncoding":
            return self

    class FakeGeneratedIds:
        def __getitem__(self, key: object) -> list[str]:
            assert key == (0, slice(3, None, None))
            return ["generated-token"]

    class FakeProcessor:
        def apply_chat_template(self, messages: object, **kwargs: object) -> FakeBatchEncoding:
            assert kwargs["enable_thinking"] is False
            assert kwargs["tokenize"] is True
            assert kwargs["return_dict"] is True
            assert kwargs["return_tensors"] == "pt"
            return FakeBatchEncoding(
                input_ids=FakeInputIds(),
                attention_mask=object(),
            )

        def decode(self, tokens: object, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            return '{"action":"search_notes","arguments":{"query":"LoRA"}}'

    class FakeModel:
        def generate(self, **kwargs: object) -> FakeGeneratedIds:
            assert "input_ids" in kwargs
            assert "attention_mask" in kwargs
            assert kwargs["do_sample"] is False
            return FakeGeneratedIds()

    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    class FakeTorch:
        def inference_mode(self) -> FakeInferenceMode:
            return FakeInferenceMode()

    model = object.__new__(TransformersDecisionModel)
    model._torch = FakeTorch()
    model._processor = FakeProcessor()
    model._model = FakeModel()
    model._device = "fake-device"
    model.max_new_tokens = 32

    action = model.decide("What was measured?", [], [])

    assert action == ToolAction("search_notes", {"query": "LoRA"})
