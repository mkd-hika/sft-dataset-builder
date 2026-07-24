"""Tests for the format-capability probe."""

from __future__ import annotations

import sft_dataset_builder.capabilities as caps
from sft_dataset_builder.capabilities import FormatStatus, probe_capabilities


def test_probe_returns_all_groups():
    statuses = probe_capabilities()
    labels = [s.label for s in statuses]
    assert any("PDF" in label for label in labels)
    assert any("Text" in label for label in labels)
    assert any("Images" in label for label in labels)
    assert any("Audio" in label for label in labels)
    assert any("Video" in label for label in labels)


def test_text_group_is_always_ready():
    text = next(s for s in probe_capabilities() if "Text" in s.label)
    assert text.ready is True
    assert text.install_hint is None
    assert ".txt" in text.extensions and ".md" in text.extensions


def test_ready_status_has_no_install_hint():
    for status in probe_capabilities():
        if status.ready:
            assert status.install_hint is None


def test_install_hint_uses_extra_when_python_missing():
    status = FormatStatus(
        label="Documents — PDF",
        extensions=(".pdf",),
        ready=False,
        extra="pdf",
        missing_python=("PyMuPDF",),
    )
    assert status.install_hint == 'pip install "sft-dataset-builder[pdf]"'


def test_install_hint_mentions_missing_binary():
    status = FormatStatus(
        label="Images (OCR)",
        extensions=(".png",),
        ready=False,
        extra="ocr",
        missing_python=("pytesseract",),
        missing_binaries=("tesseract",),
    )
    hint = status.install_hint
    assert "sft-dataset-builder[ocr]" in hint
    assert "tesseract" in hint


def test_binary_only_missing_hint_has_no_pip():
    status = FormatStatus(
        label="Images (OCR)",
        extensions=(".png",),
        ready=False,
        extra="ocr",
        missing_python=(),
        missing_binaries=("tesseract",),
    )
    assert status.install_hint == "install the 'tesseract' binary"


def test_module_installed_detects_stdlib_and_missing():
    assert caps._module_installed("json") is True
    assert caps._module_installed("a_module_that_does_not_exist_123") is False


def test_probe_reflects_a_missing_dependency(monkeypatch):
    # Force every optional module to look absent -> only pure-Python Text is ready.
    monkeypatch.setattr(caps, "_module_installed", lambda module: False)
    monkeypatch.setattr(caps, "_binary_present", lambda name: False)
    statuses = probe_capabilities()
    ready = [s for s in statuses if s.ready]
    assert len(ready) == 1
    assert "Text" in ready[0].label
    pdf = next(s for s in statuses if "PDF" in s.label)
    assert pdf.install_hint == 'pip install "sft-dataset-builder[pdf]"'
