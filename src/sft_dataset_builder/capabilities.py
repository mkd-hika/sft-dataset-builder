"""Report which input formats are ready to use vs. need an optional install.

The core pipeline is pure-Python; each media format is enabled by an optional
dependency extra (and sometimes a system binary). This module probes the
environment so a UI/CLI can show upfront which formats work — turning
mid-processing ``ImportError``s into guidance before a file is even uploaded.

>>> from sft_dataset_builder.capabilities import probe_capabilities
>>> for cap in probe_capabilities():
...     print(cap.label, "ready" if cap.ready else cap.install_hint)
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from typing import Optional

from .extractors import (
    AudioExtractor,
    DocxExtractor,
    EpubExtractor,
    HTMLExtractor,
    ImageExtractor,
    LatexExtractor,
    PDFExtractor,
    PptxExtractor,
    TabularExtractor,
    TextExtractor,
    VideoExtractor,
)
from .extractors._ocr import find_tesseract

DISTRIBUTION = "sft-dataset-builder"


@dataclass
class FormatStatus:
    """Readiness of one input-format group."""

    label: str
    extensions: tuple[str, ...]
    ready: bool
    extra: Optional[str] = None
    missing_python: tuple[str, ...] = ()
    missing_binaries: tuple[str, ...] = ()
    optional_missing: tuple[str, ...] = ()
    note: Optional[str] = None

    @property
    def install_hint(self) -> Optional[str]:
        """A short, actionable hint for what to install, or None if ready."""
        parts: list[str] = []
        if self.extra and self.missing_python:
            parts.append(f'pip install "{DISTRIBUTION}[{self.extra}]"')
        elif self.missing_python:
            parts.append("pip install " + " ".join(self.missing_python))
        for binary in self.missing_binaries:
            parts.append(f"install the '{binary}' binary")
        return " + ".join(parts) if parts else None


def _module_installed(module: str) -> bool:
    """True if ``module`` can be located (without importing it)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # A parent package is missing, or the spec is malformed.
        return False


def _binary_present(name: str) -> bool:
    if name == "tesseract":
        return find_tesseract() is not None
    return shutil.which(name) is not None


# (label, extractor, required (module, pip-name), optional (module, pip-name),
#  binaries, extra-name, note)
_SPECS: tuple[tuple, ...] = (
    ("Documents — PDF", PDFExtractor, [("fitz", "PyMuPDF")], [], [], "pdf", None),
    (
        "Documents — Word / ODT / ODP / RTF",
        DocxExtractor,
        [("docx", "python-docx"), ("odf", "odfpy")],
        [("striprtf", "striprtf")],
        [],
        "docx",
        None,
    ),
    (
        "Documents — Text / Markdown",
        TextExtractor,
        [],
        [],
        [],
        None,
        "Pure-Python — always available.",
    ),
    (
        "Web — HTML",
        HTMLExtractor,
        [],
        [("bs4", "beautifulsoup4")],
        [],
        "html",
        "Works via stdlib; the [html] extra improves fidelity.",
    ),
    (
        "Documents — LaTeX",
        LatexExtractor,
        [],
        [("pylatexenc", "pylatexenc")],
        [],
        "latex",
        "Works via a naive stripper; the [latex] extra improves fidelity.",
    ),
    (
        "Tabular — CSV / TSV",
        TabularExtractor,
        [],
        [],
        [],
        None,
        "Pure-Python — one row per sample (header-aware).",
    ),
    ("Slides — PowerPoint", PptxExtractor, [("pptx", "python-pptx")], [], [], "pptx", None),
    (
        "Books — EPUB",
        EpubExtractor,
        [("ebooklib", "EbookLib")],
        [("bs4", "beautifulsoup4")],
        [],
        "epub",
        None,
    ),
    (
        "Images (OCR)",
        ImageExtractor,
        [("pytesseract", "pytesseract"), ("PIL", "Pillow")],
        [],
        ["tesseract"],
        "ocr",
        None,
    ),
    ("Audio", AudioExtractor, [("whisper", "openai-whisper")], [], ["ffmpeg"], "audio", None),
    (
        "Video",
        VideoExtractor,
        [("cv2", "opencv-python"), ("whisper", "openai-whisper")],
        [],
        ["ffmpeg"],
        "video",
        None,
    ),
)


def probe_capabilities() -> list[FormatStatus]:
    """Inspect the environment and report readiness per input-format group."""
    statuses: list[FormatStatus] = []
    for label, extractor, required, optional, binaries, extra, note in _SPECS:
        missing_python = tuple(pip for mod, pip in required if not _module_installed(mod))
        missing_binaries = tuple(b for b in binaries if not _binary_present(b))
        optional_missing = tuple(pip for mod, pip in optional if not _module_installed(mod))
        ready = not missing_python and not missing_binaries
        statuses.append(
            FormatStatus(
                label=label,
                extensions=tuple(sorted(extractor.supported_extensions)),
                ready=ready,
                extra=extra,
                missing_python=missing_python,
                missing_binaries=missing_binaries,
                optional_missing=optional_missing,
                note=note,
            )
        )
    return statuses
