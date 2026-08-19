"""Explicit descriptions and dispatch for the four deterministic tools."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .calculator import calculate
from .constraints import check_constraints
from .notes import read_note, search_notes


@dataclass(frozen=True)
class ToolSpec:
    """A compact description of a tool and its accepted parameters."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolRegistry:
    """Describe and dispatch the project's fixed set of tools."""

    def __init__(self) -> None:
        registrations: list[tuple[ToolSpec, Callable[..., Any]]] = [
            (
                ToolSpec(
                    name="search_notes",
                    description="Search the local evidence notes for relevant documents.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                ),
                search_notes,
            ),
            (
                ToolSpec(
                    name="read_note",
                    description="Read one local evidence note by document ID.",
                    parameters={
                        "type": "object",
                        "properties": {"document_id": {"type": "string"}},
                        "required": ["document_id"],
                        "additionalProperties": False,
                    },
                ),
                read_note,
            ),
            (
                ToolSpec(
                    name="calculate",
                    description="Evaluate a simple arithmetic expression safely.",
                    parameters={
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                ),
                calculate,
            ),
            (
                ToolSpec(
                    name="check_constraints",
                    description="Check requirements against a note's metadata.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string"},
                            "requirements": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["field", "op", "value"],
                                },
                            },
                        },
                        "required": ["document_id", "requirements"],
                        "additionalProperties": False,
                    },
                ),
                check_constraints,
            ),
        ]
        self._specs = [spec for spec, _ in registrations]
        self._tools = {spec.name: function for spec, function in registrations}

    def list_specs(self) -> list[ToolSpec]:
        """Return the four tool descriptions in a stable order."""
        return list(self._specs)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a registered deterministic tool with keyword arguments."""
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a dictionary")
        try:
            tool = self._tools[tool_name]
        except KeyError as error:
            raise ValueError(f"Unknown tool: {tool_name!r}") from error
        return tool(**arguments)


def create_default_registry() -> ToolRegistry:
    """Create the registry containing the project's four tools."""
    return ToolRegistry()
