"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from sft_dataset_builder.extractors.base import ExtractedUnit


@pytest.fixture
def sample_text() -> str:
    return (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs.\n\n"
        "Sphinx of black quartz, judge my vow. "
        "How vexingly quick daft zebras jump!"
    )


@pytest.fixture
def txt_file(tmp_path: Path, sample_text: str) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(sample_text, encoding="utf-8")
    return p


def make_units(*texts: str, **common) -> list[ExtractedUnit]:
    """Convenience builder for a list of ExtractedUnit from raw strings."""
    return [ExtractedUnit(text=t, **common) for t in texts]
