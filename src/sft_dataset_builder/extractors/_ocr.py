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


def find_tesseract() -> str | None:
    """Return a usable path to the tesseract binary, or None if not found.

    Checks PATH first, then the well-known install locations above. Has no
    side effects, so it's safe to call from capability probing.
    """
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in _TESSERACT_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


def configure_tesseract(pytesseract: Any) -> None:
    """Ensure pytesseract can find the tesseract binary.

    If ``tesseract`` is already on PATH, nothing changes. Otherwise fall back to
    the well-known install locations so OCR works out of the box even when the
    binary's directory was never added to PATH (common on Windows).
    """
    if shutil.which("tesseract"):
        return
    path = find_tesseract()
    if path is not None:
        pytesseract.pytesseract.tesseract_cmd = path
