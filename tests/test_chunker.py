"""Tests for the chunker."""

from __future__ import annotations

import pytest
from sft_dataset_builder.chunker import VALID_STRATEGIES, Chunk, chunk_units
from sft_dataset_builder.extractors.base import ExtractedUnit

from .conftest import make_units


def _indices_are_sequential(chunks: list[Chunk]) -> bool:
    return [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_units_returns_empty():
    assert chunk_units([]) == []
    assert chunk_units(make_units("", "   ", "\n")) == []


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown split_strategy"):
        chunk_units(make_units("hello"), strategy="nope")


@pytest.mark.parametrize("strategy", sorted(VALID_STRATEGIES))
def test_all_strategies_produce_nonempty_sequential_chunks(strategy):
    units = make_units("First paragraph here.", "Second paragraph, a bit longer than the first.")
    chunks = chunk_units(units, strategy=strategy, chunk_size=64, overlap=8)
    assert chunks, f"{strategy} produced no chunks"
    assert all(c.text.strip() for c in chunks)
    assert _indices_are_sequential(chunks)


def test_paragraph_packs_small_units_together():
    units = make_units("alpha", "beta", "gamma")
    chunks = chunk_units(units, strategy="paragraph", chunk_size=1000)
    assert len(chunks) == 1
    assert "alpha" in chunks[0].text
    assert "gamma" in chunks[0].text


def test_paragraph_splits_when_exceeding_chunk_size():
    units = make_units("a" * 100, "b" * 100)
    chunks = chunk_units(units, strategy="paragraph", chunk_size=100, overlap=0)
    assert len(chunks) == 2


def test_sentence_strategy_splits_sentences():
    units = make_units("One sentence. Two sentence. Three sentence. Four sentence.")
    chunks = chunk_units(units, strategy="sentence", chunk_size=20, overlap=0)
    # Each ~15-char sentence should land in its own (or paired) chunk, > 1 total.
    assert len(chunks) >= 2


def test_fixed_strategy_respects_window_and_overlap():
    text = "x" * 500
    chunks = chunk_units(make_units(text), strategy="fixed", chunk_size=100, overlap=20)
    assert all(len(c.text) <= 100 for c in chunks)
    # step = chunk_size - overlap = 80; ceil(500 / 80) = 7 windows.
    assert len(chunks) == 7


def test_page_strategy_groups_by_page_number():
    units = [
        ExtractedUnit(text="page one a", page_number=1),
        ExtractedUnit(text="page one b", page_number=1),
        ExtractedUnit(text="page two", page_number=2),
    ]
    chunks = chunk_units(units, strategy="page")
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert "page one a" in chunks[0].text and "page one b" in chunks[0].text
    assert chunks[1].page_number == 2


def test_timestamp_strategy_keeps_time_span():
    units = [
        ExtractedUnit(text="hello", start_time=0.0, end_time=1.5),
        ExtractedUnit(text="world", start_time=1.5, end_time=3.0),
    ]
    chunks = chunk_units(units, strategy="timestamp", chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time == 3.0


def test_timestamp_strategy_falls_back_without_timing():
    units = make_units("no timing here", "still no timing")
    chunks = chunk_units(units, strategy="timestamp", chunk_size=1000)
    assert chunks
    assert all(c.start_time is None for c in chunks)
