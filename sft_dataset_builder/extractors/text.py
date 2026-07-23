"""Extractor for plain text and Markdown files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from .base import BaseExtractor, ExtractedUnit, ExtractionResult

_ENCODINGS_TO_TRY = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


def _read_text_file(path: Path) -> str:
    last_exc: Exception | None = None
    for enc in _ENCODINGS_TO_TRY:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError) as exc:
            last_exc = exc
            continue
    # last resort: replace undecodable bytes
    if last_exc is not None:
        return path.read_bytes().decode("utf-8", errors="replace")
    return ""


class TextExtractor(BaseExtractor):
    """Handles .txt and .md files."""

    supported_extensions: ClassVar[set[str]] = {".txt", ".md", ".markdown"}

    def extract(self, path: Path, **kwargs: Any) -> ExtractionResult:
        raw = _read_text_file(path)
        raw = raw.strip()

        # Split on blank lines to create paragraph-level units up front.
        # The chunker can regroup these however it wants later.
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        units = [
            ExtractedUnit(text=p, metadata={"paragraph_index": i}) for i, p in enumerate(paragraphs)
        ]
        if not units and raw:
            units = [ExtractedUnit(text=raw, metadata={"paragraph_index": 0})]

        return ExtractionResult(
            source_path=str(path),
            source_type=path.suffix.lower().lstrip("."),
            units=units,
            full_text=raw,
        )
