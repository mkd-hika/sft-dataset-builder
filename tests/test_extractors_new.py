"""Tests for the HTML, LaTeX, CSV/TSV, PPTX, and EPUB extractors."""

from __future__ import annotations

from pathlib import Path

import pytest

from sft_dataset_builder.extractors.html import HTMLExtractor
from sft_dataset_builder.extractors.latex import LatexExtractor, _naive_latex_strip
from sft_dataset_builder.extractors.tabular import TabularExtractor


# -- HTML ---------------------------------------------------------------------
def test_html_extracts_block_text(tmp_path: Path):
    p = tmp_path / "page.html"
    p.write_text(
        "<html><head><title>t</title><style>.x{}</style></head>"
        "<body><h1>Heading</h1><p>First paragraph.</p>"
        "<script>ignore()</script><p>Second paragraph.</p></body></html>",
        encoding="utf-8",
    )
    result = HTMLExtractor().safe_extract(p)
    assert result.success
    joined = "\n".join(u.text for u in result.units)
    assert "Heading" in joined
    assert "First paragraph." in joined
    assert "Second paragraph." in joined
    assert "ignore" not in joined  # script contents dropped


def test_html_handles_entities(tmp_path: Path):
    p = tmp_path / "e.html"
    p.write_text("<p>Fish &amp; chips &lt;yum&gt;</p>", encoding="utf-8")
    text = " ".join(u.text for u in HTMLExtractor().safe_extract(p).units)
    assert "Fish & chips" in text


# -- LaTeX --------------------------------------------------------------------
def test_latex_extraction(tmp_path: Path):
    p = tmp_path / "paper.tex"
    p.write_text(
        r"\documentclass{article}\begin{document}"
        "\n% a comment\n"
        r"\section{Intro}This is \textbf{bold} text." + "\n\n"
        r"Second paragraph here." + "\n"
        r"\end{document}",
        encoding="utf-8",
    )
    result = LatexExtractor().safe_extract(p)
    assert result.success
    joined = " ".join(u.text for u in result.units)
    assert "bold" in joined
    assert "Second paragraph" in joined


def test_naive_latex_strip_removes_commands():
    out = _naive_latex_strip(r"\section{Hi} plain \textit{ital} % comment")
    assert "Hi" in out
    assert "ital" in out
    assert "\\section" not in out
    assert "comment" not in out


# -- CSV / TSV ----------------------------------------------------------------
def test_csv_header_aware_rows(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("name,role\nAda,Engineer\nGrace,Admiral\n", encoding="utf-8")
    result = TabularExtractor().safe_extract(p)
    assert result.success
    assert len(result.units) == 2
    assert "name: Ada" in result.units[0].text
    assert "role: Engineer" in result.units[0].text
    assert "name: Grace" in result.units[1].text


def test_tsv_uses_tab_delimiter(tmp_path: Path):
    p = tmp_path / "data.tsv"
    p.write_text("a\tb\n1\t2\n", encoding="utf-8")
    result = TabularExtractor().safe_extract(p)
    assert result.units[0].text == "a: 1\nb: 2"


def test_empty_csv_reports_no_rows(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    result = TabularExtractor().safe_extract(p)
    assert result.units == []


# -- PPTX / EPUB (need optional libs) -----------------------------------------
def test_pptx_extraction(tmp_path: Path):
    pptx = pytest.importorskip("pptx")
    p = tmp_path / "deck.pptx"
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title-only layout
    slide.shapes.title.text = "Slide Title Here"
    prs.save(str(p))

    from sft_dataset_builder.extractors.pptx import PptxExtractor

    result = PptxExtractor().safe_extract(p)
    assert result.success
    assert any("Slide Title Here" in u.text for u in result.units)


def test_epub_extraction(tmp_path: Path):
    pytest.importorskip("ebooklib")
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("id1")
    book.set_title("T")
    ch = epub.EpubHtml(title="C1", file_name="c1.xhtml")
    ch.set_content(b"<html><body><h1>Chapter One</h1><p>Hello book.</p></body></html>")
    book.add_item(ch)
    book.spine = [ch]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    p = tmp_path / "book.epub"
    epub.write_epub(str(p), book)

    from sft_dataset_builder.extractors.epub import EpubExtractor

    result = EpubExtractor().safe_extract(p)
    assert result.success
    assert any("Hello book." in u.text for u in result.units)
