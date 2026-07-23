"""Tests for the formatter."""

from __future__ import annotations

import pytest

from sft_dataset_builder.chunker import Chunk
from sft_dataset_builder.formatter import VALID_FORMATS, format_chunk, format_chunks


def _chunk(text: str = "some extracted content", **kw) -> Chunk:
    return Chunk(text=text, index=kw.pop("index", 0), **kw)


def test_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown output_format"):
        format_chunk(_chunk(), output_format="nope")


@pytest.mark.parametrize("fmt", sorted(VALID_FORMATS))
def test_every_format_includes_metadata(fmt):
    sample = format_chunk(_chunk(index=3), output_format=fmt, source_file="a.pdf")
    assert sample["metadata"]["chunk_index"] == 3
    assert sample["metadata"]["source_file"] == "a.pdf"


def test_alpaca_shape():
    sample = format_chunk(_chunk("OUT"), output_format="alpaca", instruction="DO")
    assert sample["instruction"] == "DO"
    assert sample["input"] == ""
    assert sample["output"] == "OUT"


def test_sharegpt_includes_system_when_prompt_given():
    sample = format_chunk(
        _chunk("answer"), output_format="sharegpt", system_prompt="SYS", instruction="Q"
    )
    convs = sample["conversations"]
    assert convs[0] == {"from": "system", "value": "SYS"}
    assert convs[1] == {"from": "human", "value": "Q"}
    assert convs[2] == {"from": "gpt", "value": "answer"}


def test_sharegpt_omits_system_when_prompt_empty():
    sample = format_chunk(_chunk("answer"), output_format="sharegpt", system_prompt="")
    roles = [c["from"] for c in sample["conversations"]]
    assert "system" not in roles


def test_openai_shape_always_has_system():
    sample = format_chunk(_chunk("answer"), output_format="openai", system_prompt=None)
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_raw_shape():
    sample = format_chunk(_chunk("just text"), output_format="raw")
    assert sample["text"] == "just text"
    assert set(sample) == {"text", "metadata"}


@pytest.mark.parametrize("fmt", ["dpo", "orpo"])
def test_preference_formats_leave_rejected_blank(fmt):
    sample = format_chunk(_chunk("chosen text"), output_format=fmt, instruction="P")
    assert sample["prompt"] == "P"
    assert sample["chosen"] == "chosen text"
    assert sample["rejected"] == ""


def test_metadata_carries_page_and_time():
    chunk = Chunk(text="t", index=0, page_number=4, start_time=1.0, end_time=2.0)
    meta = format_chunk(chunk, output_format="raw")["metadata"]
    assert meta["page_number"] == 4
    assert meta["start_time"] == 1.0
    assert meta["end_time"] == 2.0


def test_format_chunks_maps_over_list():
    chunks = [_chunk("a", index=0), _chunk("b", index=1)]
    samples = format_chunks(chunks, output_format="raw")
    assert [s["text"] for s in samples] == ["a", "b"]
