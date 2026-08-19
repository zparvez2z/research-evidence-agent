"""Load and search the small local Markdown evidence corpus."""

from pathlib import Path
import re
import tomllib
from typing import Any


NOTES_DIRECTORY = Path(__file__).resolve().parents[3] / "data" / "notes"
_WORD_PATTERN = re.compile(r"[\w-]+")


def _parse_note(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "+++":
        raise ValueError(f"Note {path.name!r} has no TOML metadata block")

    try:
        closing_marker = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "+++"
        )
    except StopIteration as error:
        raise ValueError(f"Note {path.name!r} has an unterminated TOML metadata block") from error

    metadata = tomllib.loads("\n".join(lines[1:closing_marker]))
    document_id = metadata.get("id")
    title = metadata.get("title")
    if not isinstance(document_id, str) or not isinstance(title, str):
        raise ValueError(f"Note {path.name!r} must have string id and title metadata")

    return {
        "document_id": document_id,
        "metadata": metadata,
        "content": "\n".join(lines[closing_marker + 1 :]).strip(),
    }


def _all_notes() -> list[dict[str, Any]]:
    notes = [_parse_note(path) for path in sorted(NOTES_DIRECTORY.glob("*.md"))]
    ids = [note["document_id"] for note in notes]
    if len(ids) != len(set(ids)):
        raise ValueError("Note metadata contains duplicate document IDs")
    return notes


def read_note(document_id: str) -> dict[str, Any]:
    """Return the metadata and Markdown content for a metadata document ID."""
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-blank string")

    for note in _all_notes():
        if note["document_id"] == document_id:
            return note
    raise ValueError(f"Unknown document_id: {document_id!r}")


def _snippet(content: str, terms: list[str], maximum_length: int = 160) -> str:
    paragraphs = [paragraph.replace("\n", " ") for paragraph in content.split("\n\n")]
    matching = next(
        (paragraph for paragraph in paragraphs if any(term in paragraph.casefold() for term in terms)),
        paragraphs[0] if paragraphs else "",
    )
    if len(matching) <= maximum_length:
        return matching
    return matching[: maximum_length - 1].rstrip() + "…"


def search_notes(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search notes using case-insensitive word occurrence counts."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-blank string")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    terms = _WORD_PATTERN.findall(query.casefold())
    if not terms:
        raise ValueError("query must contain at least one searchable word")
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for note in _all_notes():
        metadata = note["metadata"]
        searchable = " ".join(
            (note["document_id"], metadata["title"], note["content"])
        ).casefold()
        score = sum(searchable.count(term) for term in terms)
        if score:
            scored.append((score, note["document_id"], note))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "document_id": note["document_id"],
            "title": note["metadata"]["title"],
            "snippet": _snippet(note["content"], terms),
        }
        for _, _, note in scored[:limit]
    ]
