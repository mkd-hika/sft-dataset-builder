"""Extractor for EPUB e-books (.epub)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from ._html import html_to_blocks
from .base import BaseExtractor, ExtractedUnit, ExtractionResult


class EpubExtractor(BaseExtractor):
    """Handles .epub: one unit per document item (chapter), text from its XHTML."""

    supported_extensions: ClassVar[set[str]] = {".epub"}

    def extract(self, path: Path, **kwargs: Any) -> ExtractionResult:
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "EbookLib is required for EPUB extraction. Install with `pip install EbookLib`."
            ) from exc

        book = epub.read_epub(str(path))
        units: list[ExtractedUnit] = []
        index = 0
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content().decode("utf-8", errors="replace")
            blocks, _ = html_to_blocks(content)
            text = "\n\n".join(blocks).strip()
            if text:
                index += 1
                units.append(
                    ExtractedUnit(
                        text=text,
                        page_number=index,
                        metadata={"item": item.get_name()},
                    )
                )

        return ExtractionResult(source_path=str(path), source_type="epub", units=units)
