"""Tests for dataset curation (clean, dedup, split, report)."""

from __future__ import annotations

import pytest

from sft_dataset_builder.curation import (
    clean_dataset,
    quality_report,
    sample_content,
    split_dataset,
)


def test_sample_content_across_formats():
    assert sample_content({"text": "raw"}) == "raw"
    assert sample_content({"instruction": "i", "input": "", "output": "out"}) == "out"
    assert sample_content({"prompt": "p", "chosen": "c", "rejected": ""}) == "c"
    assert (
        sample_content(
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
        )
        == "a"
    )
    assert (
        sample_content(
            {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "g"}]}
        )
        == "g"
    )
    assert sample_content({"weird": 1}) == ""


def _raw(text):
    return {"text": text, "metadata": {"source_file": "f.txt"}}


def test_clean_drops_empty_short_and_dedup():
    samples = [
        _raw(""),  # empty
        _raw("short"),  # < min_chars
        _raw("a" * 50),  # kept
        _raw("a" * 50),  # duplicate of previous
        _raw("B" * 50),  # kept (distinct)
    ]
    kept, report = clean_dataset(samples, min_chars=32, min_signal_ratio=0.0)
    assert report.input_count == 5
    assert report.dropped_empty == 1
    assert report.dropped_short == 1
    assert report.dropped_duplicate == 1
    assert report.kept_count == 2
    assert len(kept) == 2


def test_clean_max_chars_and_low_signal():
    samples = [
        _raw("x" * 500),
        _raw("!@#$ %^&* ()[] ---- ...."),
        _raw("Proper sentence content here."),
    ]
    _kept, report = clean_dataset(samples, min_chars=5, max_chars=100, min_signal_ratio=0.5)
    assert report.dropped_long == 1
    assert report.dropped_low_signal == 1
    assert report.kept_count == 1


def test_clean_dedup_is_normalized():
    _kept, report = clean_dataset(
        [_raw("Hello, WORLD!!"), _raw("hello world")], min_chars=1, min_signal_ratio=0.0
    )
    assert report.dropped_duplicate == 1
    assert report.kept_count == 1


def test_split_is_deterministic_and_sized():
    samples = [_raw(f"item number {i} with enough text") for i in range(100)]
    train1, val1 = split_dataset(samples, val_fraction=0.2, seed=42)
    _train2, val2 = split_dataset(samples, val_fraction=0.2, seed=42)
    assert len(val1) == 20 and len(train1) == 80
    assert [s["text"] for s in val1] == [s["text"] for s in val2]  # reproducible
    # No overlap, full coverage
    assert len({s["text"] for s in train1} & {s["text"] for s in val1}) == 0
    assert len(train1) + len(val1) == 100


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        split_dataset([_raw("x")], val_fraction=1.5)


def test_quality_report():
    samples = [_raw("a" * 10), _raw("a" * 10), _raw("b" * 30), _raw("")]
    rep = quality_report(samples).as_dict()
    assert rep["count"] == 4
    assert rep["empty"] == 1
    assert rep["duplicate_content"] == 1  # the two "aaaa..." are dupes
    assert rep["char_max"] == 30
    assert rep["by_source"]["f.txt"] == 4
