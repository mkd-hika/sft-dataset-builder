"""Command-line interface for SFT Dataset Builder.

Examples
--------
    # Build an alpaca dataset from a mix of files, write JSONL to disk:
    sft-dataset-builder doc.pdf notes.md scan.png -o dataset.jsonl

    # Stream openai-chat samples to stdout (pipe into another tool):
    sft-dataset-builder report.pdf --output-format openai | head

    # Inspect available options:
    sft-dataset-builder --list-formats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Optional, Sequence

from . import __version__
from .chunker import VALID_STRATEGIES
from .extractors import SUPPORTED_EXTENSIONS
from .formatter import VALID_FORMATS
from .pipeline import build_dataset

logger = logging.getLogger("sft_dataset_builder")

WHISPER_SIZES = ["tiny", "base", "small", "medium", "large"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sft-dataset-builder",
        description=(
            "Convert unstructured files (documents, images, audio, video) into an "
            "SFT training dataset (JSONL) — fully locally, no cloud APIs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Source files to process (any mix of supported types).",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="Write JSONL to this path. If omitted, samples are streamed to stdout.",
    )
    parser.add_argument(
        "--output-format",
        choices=sorted(VALID_FORMATS),
        default="alpaca",
        help="Training-sample shape.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=sorted(VALID_STRATEGIES),
        default="paragraph",
        help="How extracted content is chunked into samples.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2048, help="Target max characters per chunk."
    )
    parser.add_argument(
        "--overlap", type=int, default=200, help="Character overlap between consecutive chunks."
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt injected into sharegpt/openai formats.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Instruction/user-turn text (defaults to a generic instruction).",
    )

    media = parser.add_argument_group("media / extractor options")
    media.add_argument(
        "--ocr-images",
        dest="ocr_images",
        action="store_true",
        default=True,
        help="OCR embedded images in PDFs.",
    )
    media.add_argument(
        "--no-ocr-images",
        dest="ocr_images",
        action="store_false",
        help="Disable OCR of embedded PDF images.",
    )
    media.add_argument(
        "--ocr-frames",
        action="store_true",
        default=False,
        help="OCR sampled video frames (slow; off by default).",
    )
    media.add_argument(
        "--frame-interval",
        dest="frame_interval_seconds",
        type=float,
        default=10.0,
        help="Video frame sampling interval, in seconds.",
    )
    media.add_argument(
        "--model-size",
        "--whisper-model-size",
        dest="model_size",
        choices=WHISPER_SIZES,
        default="base",
        help="Whisper model size for audio/video transcription (larger = slower, more accurate).",
    )
    media.add_argument(
        "--language",
        default=None,
        help="Force a transcription language (e.g. 'en'); autodetect if omitted.",
    )

    info = parser.add_argument_group("informational")
    info.add_argument("--list-formats", action="store_true", help="List output formats and exit.")
    info.add_argument(
        "--list-strategies", action="store_true", help="List split strategies and exit."
    )
    info.add_argument(
        "--list-extensions", action="store_true", help="List supported file extensions and exit."
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (debug) logging."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``sft-dataset-builder`` console script.

    Returns a process exit code: ``0`` on success, ``1`` if every input file
    failed (so nothing could be produced), ``2`` for argument errors.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_formats:
        print("\n".join(sorted(VALID_FORMATS)))
        return 0
    if args.list_strategies:
        print("\n".join(sorted(VALID_STRATEGIES)))
        return 0
    if args.list_extensions:
        print("\n".join(sorted(SUPPORTED_EXTENSIONS)))
        return 0

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.files:
        parser.error("no input files given (see --list-extensions for supported types)")

    def on_progress(stage: str, message: str, extra: dict[str, Any]) -> None:
        print(f"[{stage}] {message}", file=sys.stderr)

    # Passed to every extractor; each ignores keys it doesn't use (all accept **kwargs).
    extractor_kwargs = {
        "ocr_images": args.ocr_images,
        "ocr_frames": args.ocr_frames,
        "frame_interval_seconds": args.frame_interval_seconds,
        "model_size": args.model_size,
        "whisper_model_size": args.model_size,
        "language": args.language,
    }

    result = build_dataset(
        files=list(args.files),
        output_format=args.output_format,
        split_strategy=args.split_strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        system_prompt=args.system_prompt,
        instruction=args.instruction,
        save_path=args.output,
        extractor_kwargs=extractor_kwargs,
        progress_callback=None if args.quiet else on_progress,
    )

    if args.output:
        if not args.quiet:
            print(
                f"Wrote {result.num_samples} sample(s) from "
                f"{result.num_successful_files}/{result.num_files} file(s) to {args.output}",
                file=sys.stderr,
            )
    else:
        for sample in result.samples:
            print(json.dumps(sample, ensure_ascii=False))

    failed = [r for r in result.file_reports if not r.success]
    if failed and not args.quiet:
        print(f"\n{len(failed)} file(s) failed:", file=sys.stderr)
        for r in failed:
            print(f"  - {r.path}: {r.error}", file=sys.stderr)

    # Non-zero only if every file failed (nothing was produced).
    if result.num_files > 0 and result.num_successful_files == 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
