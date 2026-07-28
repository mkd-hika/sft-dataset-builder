# SFT Dataset Builder 🧬

[![CI](https://github.com/mkd-hika/sft-dataset-builder/actions/workflows/ci.yml/badge.svg)](https://github.com/mkd-hika/sft-dataset-builder/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

Convert unstructured data — **documents** (PDF, DOCX, TXT, MD, RTF, ODT), **images** (OCR),
**audio** (transcription), and **video** (frame extraction + audio transcription) — into
structured **SFT (Supervised Fine-Tuning)** training datasets in JSONL format.

All heavy processing (OCR, Whisper transcription) runs **100% locally** — no cloud APIs,
no external calls, no data leaves your machine.

Use it two ways:
- **Python API** — import and call `build_dataset(...)` programmatically.
- **Streamlit UI** — drag-and-drop files, configure options, preview and download JSONL.

---

## Installation

The **core pipeline is pure-Python** with no third-party dependencies. Heavy,
format-specific libraries are opt-in via extras, so you only install what you
need. Each extractor raises a clear message if its extra is missing.

```bash
git clone <this-repo>
cd sft_dataset_builder

# Everything (all extractors + the Streamlit UI):
pip install -e ".[all]"

# ...or just the formats you care about:
pip install -e ".[pdf,docx]"      # documents
pip install -e ".[ocr]"           # image / PDF-image / video-frame OCR
pip install -e ".[audio,video]"   # Whisper transcription + video frames
pip install -e ".[ui]"            # Streamlit UI
```

| extra   | pulls in |
|---------|----------|
| `pdf`   | PyMuPDF |
| `docx`  | python-docx, odfpy, striprtf |
| `ocr`   | Pillow, pytesseract |
| `audio` | openai-whisper |
| `video` | opencv-python, openai-whisper |
| `ui`    | streamlit |
| `all`   | all of the above |
| `dev`   | pytest, ruff, mypy, build (for contributors) |

> `requirements.txt` is a convenience "install everything" pin set equivalent to
> `pip install -e ".[all]"`.

**System dependencies:**
- `tesseract-ocr` binary (for image / PDF / video-frame OCR)
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
- `ffmpeg` (used by Whisper and for extracting audio tracks from video)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`

The first time you transcribe audio/video, `openai-whisper` will download the chosen
model weights locally (cached afterward) — this still happens without any network calls
to a *cloud inference API*; it's a one-time weights download, and transcription itself
always runs on your machine.

---

## Project layout

Standard `src/` layout — the importable package lives under `src/`, tests and
tooling config at the repo root:

```
sft_dataset_builder/          # repo root
├── src/sft_dataset_builder/  # the package (see below)
├── tests/                    # pytest suite (dependency-free by default)
├── pyproject.toml            # packaging + tool config (ruff, mypy, pytest)
├── requirements.txt          # convenience "install everything" pin set
├── Makefile                  # dev shortcuts: make check / test / build
├── LICENSE · NOTICE          # Apache-2.0
└── README.md · CHANGELOG.md · CONTRIBUTING.md
```

### Package modules (`src/sft_dataset_builder/`)

```
sft_dataset_builder/
├── __init__.py          # public API: build_dataset, BuildResult, etc.
├── extractors/
│   ├── base.py           # BaseExtractor, ExtractedUnit, ExtractionResult
│   ├── text.py            # .txt, .md
│   ├── pdf.py              # .pdf (PyMuPDF + embedded image OCR)
│   ├── docx.py              # .docx, .odt, .odp, .rtf
│   ├── html.py              # .html/.htm (BeautifulSoup + stdlib fallback)
│   ├── latex.py             # .tex (pylatexenc + regex fallback)
│   ├── tabular.py           # .csv/.tsv (header-aware, one row per sample)
│   ├── pptx.py              # .pptx (python-pptx: slide text + notes)
│   ├── epub.py              # .epub (ebooklib)
│   ├── image.py             # .png/.jpg/.tiff/... (pytesseract OCR)
│   ├── audio.py             # .mp3/.wav/... (local openai-whisper)
│   └── video.py             # .mp4/.mov/... (OpenCV frames + audio track)
├── capabilities.py       # probe which formats are ready vs. need an extra/binary
├── chunker.py            # paragraph / sentence / fixed / page / timestamp splitting
├── formatter.py          # alpaca / sharegpt / openai / raw / dpo / orpo output shapes
├── pipeline.py           # orchestrates extract -> chunk -> format -> save
├── cli.py                # `sft-dataset-builder` command-line interface
└── app.py                # Streamlit UI
```

---

## Command-line usage

Installing the package puts an `sft-dataset-builder` command on your `PATH`:

```bash
# Build an alpaca dataset from a mix of files, write JSONL to disk:
sft-dataset-builder doc.pdf notes.md scan.png -o dataset.jsonl

# Stream openai-chat samples to stdout (pipe into another tool):
sft-dataset-builder report.pdf --output-format openai --quiet | head

# Tune chunking and media options:
sft-dataset-builder lecture.mp4 \
    --split-strategy timestamp \
    --model-size small \
    --ocr-frames --frame-interval 15 \
    -o lecture.jsonl

# Discover options:
sft-dataset-builder --list-formats
sft-dataset-builder --list-strategies
sft-dataset-builder --list-extensions
sft-dataset-builder --help
```

Exit codes: `0` on success, `1` if every input file failed (nothing produced),
`2` for argument errors.

---

## Python API usage

```python
from sft_dataset_builder import build_dataset

result = build_dataset(
    files=["doc.pdf", "img.png", "audio.mp3", "video.mp4"],
    output_format="alpaca",       # alpaca | sharegpt | openai | raw | dpo | orpo
    split_strategy="paragraph",   # paragraph | sentence | fixed | page | timestamp
    chunk_size=2048,
    overlap=200,
    system_prompt="You are a helpful assistant.",
    instruction="Read and understand the following content.",
    save_path="output.jsonl",
)

print(result.num_samples)          # number of training samples generated
print(result.num_successful_files) # files processed without error
print(result.total_chars)          # total characters extracted across all files

for report in result.file_reports:
    print(report.path, report.success, report.num_samples, report.error)
```

### Passing extractor-specific options

```python
result = build_dataset(
    files=["scan.pdf", "lecture.mp4"],
    output_format="sharegpt",
    split_strategy="timestamp",
    extractor_kwargs={
        "ocr_images": True,        # OCR embedded images in PDFs
        "model_size": "small",     # whisper model size: tiny/base/small/medium/large
        "ocr_frames": True,        # OCR sampled video frames
        "frame_interval_seconds": 15.0,
    },
)
```

### Progress callback (for building your own UI/CLI)

```python
def on_progress(stage, message, extra):
    print(f"[{stage}] {message}")

build_dataset(files=["a.pdf"], progress_callback=on_progress)
```

### Using extractors, chunker, formatter directly

```python
from pathlib import Path
from sft_dataset_builder.extractors import get_extractor_for
from sft_dataset_builder.chunker import chunk_units
from sft_dataset_builder.formatter import format_chunks

path = Path("report.pdf")
extractor = get_extractor_for(path)
extraction = extractor.safe_extract(path, ocr_images=True)

chunks = chunk_units(extraction.units, strategy="page")
samples = format_chunks(chunks, output_format="openai", source_file=str(path))
```

---

## Output formats

| format     | shape |
|------------|-------|
| `alpaca`   | `{"instruction", "input", "output"}` |
| `sharegpt` | `{"conversations": [{"from": "system\|human\|gpt", "value": ...}]}` |
| `openai`   | `{"messages": [{"role": "system\|user\|assistant", "content": ...}]}` |
| `raw`      | `{"text": ...}` |
| `dpo`      | `{"prompt", "chosen", "rejected": ""}` — `rejected` is left blank for you to fill in |
| `orpo`     | same shape as `dpo` |

Every sample also includes a `"metadata"` field with `source_file`, `chunk_index`, and
(when available) `page_number` / `start_time` / `end_time`.

## Split strategies

- **paragraph** — packs paragraphs/units up to `chunk_size`, minimal mid-sentence breaks.
- **sentence** — splits into sentences first, then packs up to `chunk_size`.
- **fixed** — raw fixed-width character windows with `overlap`, ignoring paragraph boundaries.
- **page** — one chunk per PDF/document page.
- **timestamp** — groups audio/video segments by time up to `chunk_size`; falls back to
  paragraph packing for sources without timing info.

---

## Streamlit UI usage

```bash
streamlit run sft_dataset_builder/app.py
# or, if installed via setup.py:
sft-dataset-builder-ui
```

Then in the browser:
1. Configure **output format**, **split strategy**, **chunk size**, **system prompt**,
   and **dataset name** in the sidebar.
2. Drag & drop any mix of supported files.
3. Click **Build dataset** — watch live progress and per-file logs.
4. Expand the **dataset preview** to sanity-check the first 3 samples.
5. Click **Download** to save the final `.jsonl`.

Errors in one file never kill the batch — failed files are reported individually with
their error message, while successfully processed files still produce samples.

---

## Supported input formats

| Category  | Extensions | Extra |
|-----------|------------|-------|
| Documents | `.pdf` | `pdf` |
| Documents | `.docx`, `.odt`, `.odp`, `.rtf` | `docx` |
| Documents | `.txt`, `.md` | — (built-in) |
| Documents | `.tex` (LaTeX) | `latex` |
| Web       | `.html`, `.htm` | `html` |
| Tabular   | `.csv`, `.tsv` | — (built-in) |
| Slides    | `.pptx` | `pptx` |
| Books     | `.epub` | `epub` |
| Images    | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`, `.gif` | `ocr` (+ tesseract) |
| Audio     | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.wma` | `audio` (+ ffmpeg) |
| Video     | `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.m4v` | `video` (+ ffmpeg) |

---

## Notes & limitations

- `dpo`/`orpo` formats produce a `chosen` sample from extracted content but leave
  `rejected` empty — you'll need to populate it (e.g. from a weaker model or manual
  curation) before using it for preference training.
- Video frame OCR is opt-in (`ocr_frames=True`) since it's the slowest option; audio
  transcription of the video's audio track is on by default.
- RTF extraction prefers `striprtf` if installed; otherwise falls back to a naive regex
  stripper with reduced fidelity.

## Curation & fine-tuning

Post-process a built dataset — deduplicate, filter, and split — and optionally
generate real instruction/answer pairs.

```bash
# clean (dedup + drop empty/short/low-signal) and 90/10 train/val split:
sft-dataset-builder corpus.pdf -o data.jsonl --clean --val-split 0.1
# -> data.train.jsonl, data.val.jsonl   (add --report stats.json for metrics)

# turn chunks into REAL instruction/answer pairs via a local LLM (Ollama/vLLM):
sft-dataset-builder corpus.pdf -o instruct.jsonl \
    --gen-qa --qa-model qwen2.5 --clean --val-split 0.1
```

From Python:
```python
from sft_dataset_builder import build_dataset, clean_dataset, split_dataset

res = build_dataset(["corpus.pdf"], output_format="raw")
kept, report = clean_dataset(res.samples, min_chars=32)   # dedup + filter
train, val = split_dataset(kept, val_fraction=0.1, seed=0)
```

> **Which output is "good for fine-tuning"?** The `raw` format is ideal for
> **continued pretraining / domain adaptation**. For **instruction tuning**, use
> `--gen-qa` (or `generate_instruction_samples`) to synthesize genuine
> instruction/answer pairs from the extracted text, then clean + split. See
> [`examples/`](examples/) for an end-to-end script and a TRL LoRA fine-tuning
> example.

## Development

```bash
# Editable install with all extras + dev tooling:
pip install -e ".[all,dev]"
pre-commit install     # optional: run ruff + mypy on every commit

# Quality gates (all run in CI):
make check             # = lint + typecheck + test
python -m ruff check . # lint
python -m ruff format .# format
python -m mypy         # type-check
python -m pytest       # tests
python -m pytest --cov=sft_dataset_builder --cov-report=term-missing
```

The default test suite is dependency-free (it exercises the pure-Python core:
chunker, formatter, pipeline, text extractor, registry, and CLI). Tests that
need optional binaries are skipped unless those binaries/packages are present.
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a new extractor.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

