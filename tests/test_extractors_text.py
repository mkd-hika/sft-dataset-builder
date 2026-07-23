"""Tests for the plain-text / Markdown extractor and the base extractor contract."""

from __future__ import annotations

from pathlib import Path

from sft_dataset_builder.extractors.text import TextExtractor


def test_splits_into_paragraph_units(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text("Para one.\n\nPara two.\n\nPara three.", encoding="utf-8")
    result = TextExtractor().safe_extract(p)
    assert result.success
    assert len(result.units) == 3
    assert result.units[0].text == "Para one."
    assert result.char_count > 0


def test_markdown_extension_is_handled():
    ext = TextExtractor()
    assert ext.can_handle(Path("notes.md"))
    assert ext.can_handle(Path("notes.markdown"))
    assert ext.can_handle(Path("NOTES.TXT"))  # case-insensitive
    assert not ext.can_handle(Path("image.png"))


def test_empty_file_yields_no_units(tmp_path: Path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n\n  ", encoding="utf-8")
    result = TextExtractor().safe_extract(p)
    assert result.success
    assert result.units == []


def test_single_block_without_blank_lines(tmp_path: Path):
    p = tmp_path / "oneblock.txt"
    p.write_text("just one line of content", encoding="utf-8")
    result = TextExtractor().safe_extract(p)
    assert len(result.units) == 1


def test_non_utf8_falls_back_to_latin1(tmp_path: Path):
    p = tmp_path / "latin.txt"
    # 0xe9 is 'é' in latin-1/cp1252 but invalid as standalone UTF-8.
    p.write_bytes(b"caf\xe9 au lait")
    result = TextExtractor().safe_extract(p)
    assert result.success
    assert "caf" in result.units[0].text


def test_safe_extract_finalizes_char_count(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text("hello world", encoding="utf-8")
    result = TextExtractor().safe_extract(p)
    assert result.char_count == len(result.full_text)
