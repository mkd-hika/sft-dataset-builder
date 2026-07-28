# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0]

### Added
- **Dataset curation** (`sft_dataset_builder.curation`): `clean_dataset`
  (drop empty / too-short / too-long / low-signal content and near-duplicates),
  `split_dataset` (deterministic train/validation split), and `quality_report`.
  Exposed via new CLI flags (`--clean`, `--min-chars`, `--max-chars`,
  `--no-dedup`, `--min-signal`, `--val-split`, `--seed`, `--report`) and a
  "Post-processing" section in the Streamlit UI (cleaning report + train/val
  downloads).
- **Instruction generation** (`sft_dataset_builder.instruction_gen`): turn
  extracted chunks into real instruction/answer pairs via a pluggable LLM
  (`generate_instruction_samples`), with a stdlib OpenAI-compatible client
  (`openai_compatible_client`, local Ollama/vLLM by default). CLI: `--gen-qa`
  with `--qa-base-url` / `--qa-model` / `--qa-api-key` / `--qa-max-items`.
- `examples/`: `generate_instructions.py` (extract → LLM instruction pairs →
  clean → split) and `finetune_trl.py` (LoRA SFT on the JSONL), plus a README.
- New input formats and extractors: **HTML** (`.html/.htm`, `[html]` — with a
  stdlib fallback), **LaTeX** (`.tex`, `[latex]` — with a regex fallback),
  **PowerPoint** (`.pptx`, `[pptx]`), **EPUB** (`.epub`, `[epub]`), and
  **CSV/TSV** (`.csv/.tsv`, built-in, header-aware one-row-per-sample).
  `.odp` is now handled alongside `.odt` (no new dependency).
- Format-readiness probe (`sft_dataset_builder.capabilities.probe_capabilities`)
  that reports which input-format groups are usable vs. need an optional extra
  or system binary, with an actionable install hint per group.
- Streamlit UI now shows a "Supported inputs" panel (ready ✅ / needs-install ⚠️)
  above the uploader, so missing dependencies are surfaced before upload instead
  of failing mid-build.
- Tesseract binary auto-detection (`extractors/_ocr.py`) so image / PDF-image /
  video-frame OCR works without manual PATH setup on Windows/macOS/Linux.

## [0.2.0]

### Added
- Command-line interface: `sft-dataset-builder` wraps `build_dataset`, supports
  streaming JSONL to stdout or writing with `-o/--output`, and exposes all
  pipeline and extractor options (`--list-formats`, `--list-strategies`,
  `--list-extensions`, `--version`).
- Modern packaging via `pyproject.toml` (PEP 621) with **optional dependency
  extras** per format group (`pdf`, `docx`, `ocr`, `audio`, `video`, `ui`,
  plus `all` and `dev`). The core pipeline is now pure-Python with no required
  third-party dependencies.
- `LICENSE` (Apache-2.0) and `NOTICE`, `py.typed` marker (PEP 561 typed
  package), `MANIFEST.in`.
- Test suite under `tests/` covering the chunker, formatter, pipeline, text
  extractor, extractor registry, and CLI.
- Tooling configuration: Ruff (lint + format), Mypy, pytest, coverage,
  pre-commit hooks, and `.editorconfig`.
- GitHub Actions CI: lint, type-check, test matrix (Python 3.9–3.12 on
  Linux/macOS/Windows), and distribution build with `twine check`.
- `CONTRIBUTING.md`, `CHANGELOG.md`, and a `Makefile` with common dev targets.

### Changed
- Heavy dependencies (PyMuPDF, python-docx, Pillow, pytesseract, opencv-python,
  openai-whisper, streamlit) moved from hard requirements to opt-in extras;
  install everything with `pip install "sft-dataset-builder[all]"`.
- Removed the legacy `setup.py`; the build is now fully driven by
  `pyproject.toml`.

### Removed
- Dropped unused `numpy` / `tqdm` direct dependencies (they are pulled in
  transitively by the media extras only when needed).

## [0.1.0]

### Added
- Initial release: extractors for documents (PDF, DOCX, ODT, RTF, TXT, MD),
  images (OCR), audio (Whisper transcription), and video (frames + audio).
- Configurable chunker (paragraph, sentence, fixed, page, timestamp) and
  formatter (alpaca, sharegpt, openai, raw, dpo, orpo).
- `build_dataset` pipeline and a Streamlit UI.

[Unreleased]: https://github.com/mkd-hika/sft-dataset-builder/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mkd-hika/sft-dataset-builder/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mkd-hika/sft-dataset-builder/releases/tag/v0.1.0
