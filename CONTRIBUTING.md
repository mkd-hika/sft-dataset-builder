# Contributing

Thanks for your interest in improving **SFT Dataset Builder**! This guide covers
local setup, the quality bar, and how to add support for new file types.

## Development setup

```bash
git clone <this-repo>
cd sft_dataset_builder

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# Editable install with all runtime extras + dev tools:
pip install -e ".[all,dev]"

# Install the git hooks (runs ruff + mypy on commit):
pre-commit install
```

> The core pipeline is pure-Python and needs no third-party packages. The
> `[all]` extra pulls in the media dependencies (PyMuPDF, Pillow, pytesseract,
> openai-whisper, opencv-python, ...). OCR also needs the `tesseract` binary and
> audio/video need `ffmpeg` on your `PATH`.

## Quality checks

All of these run in CI and must pass:

```bash
make check          # lint + typecheck + test
# or individually:
python -m ruff check .          # lint
python -m ruff format .         # format
python -m mypy                  # type-check
python -m pytest                # tests
python -m pytest --cov=sft_dataset_builder --cov-report=term-missing
```

- Code is formatted and linted with **Ruff** (config in `pyproject.toml`).
- Public functions and dataclasses carry type hints; the package ships a
  `py.typed` marker, so keep it type-clean under **Mypy**.
- New behavior needs tests. Tests that require optional system binaries should
  use `pytest.importorskip(...)` or the `requires_tesseract` / `requires_ffmpeg`
  / `requires_whisper` markers so the default suite stays dependency-free.

## Adding a new extractor

1. Create `src/sft_dataset_builder/extractors/<name>.py` subclassing `BaseExtractor`.
2. Set `supported_extensions` and implement `extract(self, path, **kwargs)`,
   returning an `ExtractionResult` built from `ExtractedUnit`s. Import heavy,
   optional dependencies **inside** `extract` and raise a clear `ImportError`
   telling the user which extra to install — mirror the existing extractors.
3. Register an instance in `ALL_EXTRACTORS` in `extractors/__init__.py`.
4. Add its dependency to a suitable extra (or a new one) in `pyproject.toml` and
   to `requirements.txt`.
5. Add tests and update the README's supported-formats table.

## Commit / PR conventions

- Keep PRs focused; update `CHANGELOG.md` under `[Unreleased]`.
- Describe user-facing changes and note any new optional dependencies.
