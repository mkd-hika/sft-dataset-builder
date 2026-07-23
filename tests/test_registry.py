"""Tests for the extractor registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from sft_dataset_builder.extractors import (
    SUPPORTED_EXTENSIONS,
    DocxExtractor,
    ImageExtractor,
    PDFExtractor,
    TextExtractor,
    get_extractor_for,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.txt", TextExtractor),
        ("a.md", TextExtractor),
        ("a.pdf", PDFExtractor),
        ("a.docx", DocxExtractor),
        ("a.odt", DocxExtractor),
        ("a.rtf", DocxExtractor),
        ("a.png", ImageExtractor),
        ("a.JPG", ImageExtractor),  # case-insensitive
    ],
)
def test_get_extractor_for_known_types(filename, expected):
    assert isinstance(get_extractor_for(Path(filename)), expected)


def test_unsupported_type_returns_none():
    assert get_extractor_for(Path("archive.zip")) is None
    assert get_extractor_for(Path("noext")) is None


def test_supported_extensions_cover_core_types():
    for ext in (".pdf", ".txt", ".md", ".png", ".mp3", ".mp4", ".docx"):
        assert ext in SUPPORTED_EXTENSIONS


def test_supported_extensions_are_lowercase_and_dotted():
    for ext in SUPPORTED_EXTENSIONS:
        assert ext.startswith(".")
        assert ext == ext.lower()
