"""Base classes shared by all extractors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

logger = logging.getLogger("sft_dataset_builder")


@dataclass
class ExtractedUnit:
    """A single unit of extracted content (e.g. a page, a segment, a frame).

    Units are the atomic pieces produced by extractors. The chunker later
    regroups/splits them according to the chosen split strategy.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None  # seconds, for audio/video
    end_time: Optional[float] = None
    page_number: Optional[int] = None


@dataclass
class ExtractionResult:
    """Result of extracting a single source file."""

    source_path: str
    source_type: str
    units: list[ExtractedUnit] = field(default_factory=list)
    full_text: str = ""
    char_count: int = 0
    success: bool = True
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def finalize(self) -> "ExtractionResult":
        if not self.full_text:
            self.full_text = "\n\n".join(u.text for u in self.units if u.text and u.text.strip())
        self.char_count = len(self.full_text)
        return self


class BaseExtractor(ABC):
    """Interface every extractor must implement."""

    #: lower-case file extensions this extractor supports, e.g. {".pdf"}
    supported_extensions: ClassVar[set[str]] = set()

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def extract(self, path: Path, **kwargs: Any) -> ExtractionResult:
        """Extract structured content from `path`. Must not raise on
        recoverable errors -- return an ExtractionResult with success=False
        and `error` set instead, so batch pipelines can continue.
        """
        raise NotImplementedError

    def safe_extract(self, path: Path, **kwargs: Any) -> ExtractionResult:
        try:
            result = self.extract(path, **kwargs)
            return result.finalize()
        except Exception as exc:
            # Intentional broad catch: an extractor must never abort a batch;
            # failures are reported via ExtractionResult(success=False).
            logger.exception("Extraction failed for %s", path)
            return ExtractionResult(
                source_path=str(path),
                source_type=path.suffix.lower().lstrip("."),
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
