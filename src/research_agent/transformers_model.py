"""Optional Hugging Face Transformers decision-model adapter."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .actions import Action, FinalAction, ToolAction
from .state import ToolObservation
from .tools.registry import ToolSpec


SYSTEM_INSTRUCTION = """You are the decision component of a Research Evidence Agent.
At each call, select exactly ONE next action: call one available tool or provide a final answer.
Return exactly one JSON object.
Tool action: {"action":"<tool name>","arguments":{}}
Final action: {"action":"final","answer":"<user-facing answer>","evidence_ids":["<document id>"]}
Use only the supplied tool specifications and previous observations.
search_notes discovers candidate document IDs; its snippets are discovery information, not authoritative final-answer evidence. After finding a relevant result, normally call read_note instead of repeating the same search.
Do not repeat an identical successful tool call with identical arguments unless new information genuinely requires it.
Before a final answer, every cited source must have been successfully read with read_note. Do not invent document IDs.
For unavailable or missing measurements, search for relevant evidence, read the document explaining the limitation, then return a grounded insufficient-evidence answer.
For constraint questions, use check_constraints only on experiment documents. Use exact metadata field names and exact supported operators from its tool specification; do not call it on evaluation-protocol. Do not invent aliases such as "F1", "latency", or "local inference" when the specified fields are "f1", "latency_ms", and "local_inference".
Do not output analysis, rationale, scratchpad, markdown, code fences, or text outside the action JSON."""


def tool_spec_to_dict(spec: ToolSpec) -> dict[str, Any]:
    """Convert one tool specification to explicit JSON-compatible fields."""
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.parameters,
    }


def observation_to_dict(observation: ToolObservation) -> dict[str, Any]:
    """Convert one observable tool outcome to explicit prompt fields."""
    item: dict[str, Any] = {
        "step": observation.step,
        "tool": observation.tool,
        "arguments": observation.arguments,
    }
    if observation.error is None:
        item["result"] = observation.result
    else:
        item["error"] = observation.error
    return item


def build_user_context(
    question: str,
    observations: Sequence[ToolObservation],
    tools: Sequence[ToolSpec],
) -> str:
    """Build deterministic, human-readable JSON from observable context only."""
    context = {
        "question": question,
        "tool_specifications": [tool_spec_to_dict(spec) for spec in tools],
        "previous_tool_observations": [
            observation_to_dict(observation) for observation in observations
        ],
    }
    return json.dumps(context, indent=2, sort_keys=True)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        return stripped[len("```json") : -len("```")].strip()
    return stripped


def _parse_qwen_action(value: dict[str, Any]) -> Action | None:
    """Parse the narrow action shape observed from Qwen3.5, if present."""
    if "action" not in value or "type" in value:
        return None

    action_name = value.get("action")
    if not isinstance(action_name, str) or not action_name.strip():
        raise ValueError("Model action must be a non-blank string")

    if action_name == "final":
        expected = {"action", "answer", "evidence_ids"}
        if set(value) != expected:
            raise ValueError(
                "Final action must contain exactly: action, answer, evidence_ids"
            )
        if not isinstance(value["answer"], str) or not value["answer"].strip():
            raise ValueError("Final action answer must be a non-blank string")
        evidence_ids = value["evidence_ids"]
        if not isinstance(evidence_ids, list) or any(
            not isinstance(item, str) or not item.strip() for item in evidence_ids
        ):
            raise ValueError("Final action evidence_ids must be an array of strings")
        return FinalAction(answer=value["answer"], evidence_ids=evidence_ids)

    expected = {"action", "arguments"}
    if set(value) != expected:
        raise ValueError("Tool action must contain exactly: action, arguments")
    if not isinstance(value["arguments"], dict):
        raise ValueError("Tool action arguments must be an object")
    return ToolAction(tool=action_name, arguments=value["arguments"])


def parse_model_action(text: str) -> Action:
    """Parse one strict supported model JSON object into an existing action dataclass."""
    try:
        value = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid model action JSON: {error.msg}") from error

    if not isinstance(value, dict):
        raise ValueError("Model action must be a JSON object")

    qwen_action = _parse_qwen_action(value)
    if qwen_action is not None:
        return qwen_action

    action_type = value.get("type")
    if action_type == "tool":
        expected = {"type", "tool", "arguments"}
        if set(value) != expected:
            raise ValueError("Tool action must contain exactly: type, tool, arguments")
        if not isinstance(value["tool"], str) or not value["tool"].strip():
            raise ValueError("Tool action tool must be a non-blank string")
        if not isinstance(value["arguments"], dict):
            raise ValueError("Tool action arguments must be an object")
        return ToolAction(tool=value["tool"], arguments=value["arguments"])

    if action_type == "final":
        expected = {"type", "answer", "evidence_ids"}
        if set(value) != expected:
            raise ValueError(
                "Final action must contain exactly: type, answer, evidence_ids"
            )
        if not isinstance(value["answer"], str) or not value["answer"].strip():
            raise ValueError("Final action answer must be a non-blank string")
        evidence_ids = value["evidence_ids"]
        if not isinstance(evidence_ids, list) or any(
            not isinstance(item, str) or not item.strip() for item in evidence_ids
        ):
            raise ValueError("Final action evidence_ids must be an array of strings")
        return FinalAction(answer=value["answer"], evidence_ids=evidence_ids)

    raise ValueError(f"Unknown model action type: {action_type!r}")


def _import_model_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "TransformersDecisionModel requires the optional Colab/model "
            "dependencies. Follow the Transformers-from-main and editable "
            "installation commands in demo_colab.ipynb."
        ) from error
    return torch, AutoProcessor, AutoModelForMultimodalLM


class TransformersDecisionModel:
    """Select structured actions with a local Hugging Face multimodal model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-2B",
        max_new_tokens: int = 256,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-blank string")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")

        torch, processor_class, model_class = _import_model_dependencies()
        self._torch = torch
        self._processor = processor_class.from_pretrained(model_name)
        model_options = {"dtype": torch.float16} if torch.cuda.is_available() else {}
        self._model = model_class.from_pretrained(model_name, **model_options)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self.max_new_tokens = max_new_tokens

    def decide(
        self,
        question: str,
        observations: Sequence[ToolObservation],
        tools: Sequence[ToolSpec],
    ) -> Action:
        """Generate and parse exactly one next action without executing it."""
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": build_user_context(question, observations, tools),
            },
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)
        prompt_length = inputs["input_ids"].shape[-1]
        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        new_tokens = generated_ids[0, prompt_length:]
        response = self._processor.decode(new_tokens, skip_special_tokens=True)
        return parse_model_action(response)
