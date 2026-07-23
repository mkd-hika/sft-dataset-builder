"""Image extractor: OCR via pytesseract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from .base import BaseExtractor, ExtractedUnit, ExtractionResult


class ImageExtractor(BaseExtractor):
    """Handles common raster image formats via OCR."""

    supported_extensions: ClassVar[set[str]] = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".gif",
    }

    def extract(
        self,
        path: Path,
        lang: str = "eng",
        preprocess: bool = True,
        **kwargs: Any,
    ) -> ExtractionResult:
        try:
            import pytesseract
            from PIL import Image, ImageOps
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pytesseract and Pillow are required for image OCR. "
                "Install with `pip install pytesseract Pillow` and ensure the "
                "tesseract binary is installed on your system."
            ) from exc

        warnings: list[str] = []
        # Image.open returns an ImageFile; annotate as the base Image so the
        # later .convert()/ImageOps reassignments type-check cleanly.
        image: Image.Image = Image.open(path)
        image = image.convert("RGB")

        if preprocess:
            try:
                gray = ImageOps.grayscale(image)
                image = ImageOps.autocontrast(gray)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"preprocessing failed, using original image ({exc})")
                image = Image.open(path).convert("RGB")

        try:
            text = pytesseract.image_to_string(image, lang=lang).strip()
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "The tesseract binary was not found on this system. "
                "Install it (e.g. `apt install tesseract-ocr` or `brew install tesseract`)."
            ) from exc

        units = []
        if text:
            units.append(
                ExtractedUnit(
                    text=text,
                    metadata={"width": image.width, "height": image.height, "ocr_lang": lang},
                )
            )
        else:
            warnings.append("OCR produced no text for this image.")

        return ExtractionResult(
            source_path=str(path),
            source_type=path.suffix.lower().lstrip("."),
            units=units,
            warnings=warnings,
        )
