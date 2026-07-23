"""PDF extractor built on PyMuPDF (fitz).

Extracts per-page text, and optionally OCRs embedded raster images (scanned
pages, figures with text, etc.) using pytesseract, merging the OCR text into
the page's unit.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, ClassVar

from ._ocr import configure_tesseract
from .base import BaseExtractor, ExtractedUnit, ExtractionResult

logger = logging.getLogger("sft_dataset_builder")


class PDFExtractor(BaseExtractor):
    """Handles .pdf files."""

    supported_extensions: ClassVar[set[str]] = {".pdf"}

    def extract(
        self,
        path: Path,
        ocr_images: bool = True,
        ocr_min_image_size: int = 64,
        **kwargs: Any,
    ) -> ExtractionResult:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyMuPDF is required for PDF extraction. Install with `pip install PyMuPDF`."
            ) from exc

        warnings: list[str] = []
        units: list[ExtractedUnit] = []

        doc = fitz.open(str(path))
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_text = page.get_text("text").strip()

                ocr_text = ""
                if ocr_images:
                    ocr_text = self._ocr_page_images(
                        doc, page, ocr_min_image_size, warnings, page_index
                    )

                combined = page_text
                if ocr_text:
                    combined = (page_text + "\n\n[Image OCR]\n" + ocr_text).strip()

                if combined:
                    units.append(
                        ExtractedUnit(
                            text=combined,
                            page_number=page_index + 1,
                            metadata={
                                "has_embedded_ocr": bool(ocr_text),
                                "page_count": len(doc),
                            },
                        )
                    )
        finally:
            doc.close()

        return ExtractionResult(
            source_path=str(path),
            source_type="pdf",
            units=units,
            warnings=warnings,
        )

    @staticmethod
    def _ocr_page_images(
        doc: Any,
        page: Any,
        min_size: int,
        warnings: list[str],
        page_index: int,
    ) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            warnings.append(
                f"page {page_index + 1}: skipped image OCR (pytesseract/Pillow not installed)"
            )
            return ""

        configure_tesseract(pytesseract)
        texts: list[str] = []
        try:
            image_list = page.get_images(full=True)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"page {page_index + 1}: could not enumerate images ({exc})")
            return ""

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes))
                if pil_image.width < min_size or pil_image.height < min_size:
                    continue
                text = pytesseract.image_to_string(pil_image).strip()
                if text:
                    texts.append(text)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"page {page_index + 1}, image {img_index}: OCR failed ({exc})")
                continue

        return "\n".join(texts)
