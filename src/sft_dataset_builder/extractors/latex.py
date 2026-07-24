"""Extractor for LaTeX source files (.tex)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

from .base import BaseExtractor, ExtractedUnit, ExtractionResult
from .text import _read_text_file


class LatexExtractor(BaseExtractor):
    """Handles .tex via pylatexenc, falling back to a naive stripper."""

    supported_extensions: ClassVar[set[str]] = {".tex"}

    def extract(self, path: Path, **kwargs: Any) -> ExtractionResult:
        raw = _read_text_file(path)
        warnings: list[str] = []
        try:
            from pylatexenc.latex2text import LatexNodes2Text

            plain = LatexNodes2Text().latex_to_text(raw)
        except ImportError:
            warnings.append(
                "pylatexenc not installed; used a naive LaTeX stripper "
                "(install the [latex] extra for higher-fidelity conversion)."
            )
            plain = _naive_latex_strip(raw)

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", plain) if p.strip()]
        units = [
            ExtractedUnit(text=p, metadata={"paragraph_index": i}) for i, p in enumerate(paragraphs)
        ]
        return ExtractionResult(
            source_path=str(path),
            source_type="tex",
            units=units,
            warnings=warnings,
        )


def _naive_latex_strip(raw: str) -> str:
    """Rough LaTeX -> text fallback when pylatexenc isn't available."""
    text = re.sub(r"(?<!\\)%.*", "", raw)  # comments
    text = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", "", text)  # environment delimiters
    # commands with an optional [..] arg and/or a single {..} arg
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
