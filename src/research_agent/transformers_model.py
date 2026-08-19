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
Return ONLY one JSON object.
Tool action: {"type":"tool","tool":"<tool name>","arguments":{}}
Final action: {"type":"final","answer":"<user-facing answer>","evidence_ids":["<document id>"]}
Use only tools in the supplied tool specifications and use previous observations.
Do not invent document IDs. Final answers must cite IDs successfully read with read_note.
If requested information was not measured or is unavailable, say the evidence is insufficient.
Do not output analysis, rationale, markdown, code fences, or text outside the JSON object."""


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


def parse_model_action(text: str) -> Action:
    """Parse one strict model JSON object into an existing action dataclass."""
    try:
        value = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid model action JSON: {error.msg}") from error

    if not isinstance(value, dict):
        raise ValueError("Model action must be a JSON object")

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
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "TransformersDecisionModel requires the optional Colab/model "
            "dependencies. Install them with: pip install -e '.[colab]'"
        ) from error
    return torch, AutoTokenizer, AutoModelForCausalLM


class TransformersDecisionModel:
    """Select structured actions with a local Hugging Face causal language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
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

        torch, tokenizer_class, model_class = _import_model_dependencies()
        self._torch = torch
        self._tokenizer = tokenizer_class.from_pretrained(model_name)
        model_options = {"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
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
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = generated_ids[0, input_ids.shape[-1] :]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_model_action(response)
