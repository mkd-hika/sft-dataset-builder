"""Extractor registry: maps file extensions to extractor instances."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .audio import AudioExtractor
from .base import BaseExtractor, ExtractedUnit, ExtractionResult
from .docx import DocxExtractor
from .epub import EpubExtractor
from .html import HTMLExtractor
from .image import ImageExtractor
from .latex import LatexExtractor
from .pdf import PDFExtractor
from .pptx import PptxExtractor
from .tabular import TabularExtractor
from .text import TextExtractor
from .video import VideoExtractor

__all__ = [
    "ALL_EXTRACTORS",
    "SUPPORTED_EXTENSIONS",
    "AudioExtractor",
    "BaseExtractor",
    "DocxExtractor",
    "EpubExtractor",
    "ExtractedUnit",
    "ExtractionResult",
    "HTMLExtractor",
    "ImageExtractor",
    "LatexExtractor",
    "PDFExtractor",
    "PptxExtractor",
    "TabularExtractor",
    "TextExtractor",
    "VideoExtractor",
    "get_extractor_for",
]

ALL_EXTRACTORS: list[BaseExtractor] = [
    TextExtractor(),
    PDFExtractor(),
    DocxExtractor(),
    HTMLExtractor(),
    LatexExtractor(),
    TabularExtractor(),
    PptxExtractor(),
    EpubExtractor(),
    ImageExtractor(),
    AudioExtractor(),
    VideoExtractor(),
]

SUPPORTED_EXTENSIONS: set[str] = set().union(*(e.supported_extensions for e in ALL_EXTRACTORS))


def get_extractor_for(path: Path) -> Optional[BaseExtractor]:
    """Return the first extractor able to handle `path`, or None."""
    for extractor in ALL_EXTRACTORS:
        if extractor.can_handle(path):
            return extractor
    return None
