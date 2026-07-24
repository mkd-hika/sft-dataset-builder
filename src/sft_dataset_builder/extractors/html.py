"""Extractor for HTML pages (.html/.htm)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from ._html import html_to_blocks
from .base import BaseExtractor, ExtractedUnit, ExtractionResult
from .text import _read_text_file


class HTMLExtractor(BaseExtractor):
    """Handles .html/.htm via BeautifulSoup, with a stdlib parser fallback."""

    supported_extensions: ClassVar[set[str]] = {".html", ".htm"}

    def extract(self, path: Path, **kwargs: Any) -> ExtractionResult:
        raw = _read_text_file(path)
        blocks, used_bs4 = html_to_blocks(raw)

        warnings: list[str] = []
        if not used_bs4:
            warnings.append(
                "beautifulsoup4 not installed; used a lower-fidelity stdlib HTML "
                "parser (install the [html] extra for better results)."
            )

        units = [
            ExtractedUnit(text=block, metadata={"block_index": i}) for i, block in enumerate(blocks)
        ]
        return ExtractionResult(
            source_path=str(path),
            source_type="html",
            units=units,
            warnings=warnings,
        )
