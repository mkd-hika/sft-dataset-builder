"""Shared helpers for the OCR-based extractors (image, PDF images, video frames)."""

from __future__ import annotations

import os
import shutil
from typing import Any

# Standard install locations for the Tesseract binary that are frequently NOT on
# PATH — notably the Windows UB-Mannheim installer and a common per-user install.
_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/usr/bin/tesseract",
)


def configure_tesseract(pytesseract: Any) -> None:
    """Ensure pytesseract can find the tesseract binary.

    If ``tesseract`` is already on PATH, nothing changes. Otherwise fall back to
    the well-known install locations above so OCR works out of the box even when
    the binary's directory was never added to PATH (common on Windows).
    """
    if shutil.which("tesseract"):
        return
    for candidate in _TESSERACT_CANDIDATES:
        if os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return
