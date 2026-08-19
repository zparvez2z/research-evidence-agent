import pytest

from research_agent.tools import read_note, search_notes


def test_search_notes_finds_expected_result() -> None:
    results = search_notes("quantized low-latency")
    assert results[0]["document_id"] == "quantized-small"
    assert set(results[0]) == {"document_id", "title", "snippet"}


def test_search_notes_is_case_insensitive() -> None:
    assert search_notes("LORA") == search_notes("lora")


def test_search_notes_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="non-blank"):
        search_notes("  ")


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_search_notes_rejects_invalid_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        search_notes("model", limit)  # type: ignore[arg-type]


def test_search_notes_orders_ties_by_document_id() -> None:
    result_ids = [result["document_id"] for result in search_notes("small", limit=10)]
    assert result_ids == ["lora-small", "quantized-small"]


def test_read_note_returns_correct_note() -> None:
    note = read_note("distilled-hybrid")
    assert note["document_id"] == "distilled-hybrid"
    assert note["metadata"]["f1"] == 0.73
    assert "distilled hybrid model" in note["content"].lower()


def test_read_note_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="Unknown document_id"):
        read_note("not-a-note")
