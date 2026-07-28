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
from pathlib import Path
from typing import Any, Optional, Sequence

from . import __version__
from .chunker import VALID_STRATEGIES
from .curation import clean_dataset, quality_report, split_dataset
from .extractors import SUPPORTED_EXTENSIONS
from .formatter import VALID_FORMATS
from .instruction_gen import generate_instruction_samples, openai_compatible_client
from .pipeline import build_dataset, save_jsonl

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

    post = parser.add_argument_group("post-processing (curation)")
    post.add_argument(
        "--clean",
        action="store_true",
        help="Clean the dataset: drop empty / too-short / low-signal content and dedup.",
    )
    post.add_argument("--min-chars", type=int, default=32, help="Min content chars (with --clean).")
    post.add_argument(
        "--max-chars", type=int, default=None, help="Max content chars (with --clean)."
    )
    post.add_argument(
        "--no-dedup",
        dest="dedup",
        action="store_false",
        default=True,
        help="Disable near-duplicate removal.",
    )
    post.add_argument(
        "--min-signal",
        type=float,
        default=0.2,
        help="Min alphanumeric ratio to keep, filters OCR noise (with --clean).",
    )
    post.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Validation fraction; writes <out>.train.jsonl and <out>.val.jsonl.",
    )
    post.add_argument("--seed", type=int, default=0, help="Shuffle seed for --val-split.")
    post.add_argument(
        "--report", metavar="PATH", default=None, help="Write a JSON quality report to PATH."
    )

    qa = parser.add_argument_group("instruction generation (LLM)")
    qa.add_argument(
        "--gen-qa",
        action="store_true",
        help="Generate real instruction/answer pairs from chunks via an LLM.",
    )
    qa.add_argument(
        "--qa-base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible endpoint for --gen-qa (default: local Ollama).",
    )
    qa.add_argument("--qa-model", default="qwen2.5", help="Model name for --gen-qa.")
    qa.add_argument("--qa-api-key", default="", help="API key for --gen-qa, if required.")
    qa.add_argument(
        "--qa-max-items", type=int, default=None, help="Cap the number of generated pairs."
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

    post_processing = bool(args.clean or args.val_split > 0 or args.report or args.gen_qa)

    result = build_dataset(
        files=list(args.files),
        output_format="raw" if args.gen_qa else args.output_format,
        split_strategy=args.split_strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        system_prompt=args.system_prompt,
        instruction=args.instruction,
        # When post-processing, save after the extra steps rather than the raw build.
        save_path=None if post_processing else args.output,
        extractor_kwargs=extractor_kwargs,
        progress_callback=None if args.quiet else on_progress,
    )

    samples = result.samples

    if args.gen_qa:
        complete = openai_compatible_client(
            base_url=args.qa_base_url, model=args.qa_model, api_key=args.qa_api_key
        )
        samples = generate_instruction_samples(
            samples,
            complete,
            output_format=args.output_format,
            system_prompt=args.system_prompt,
            max_items=args.qa_max_items,
            on_error=None
            if args.quiet
            else (lambda i, exc: print(f"[gen-qa] chunk {i}: {exc}", file=sys.stderr)),
        )
        if not args.quiet:
            print(f"[gen-qa] generated {len(samples)} instruction pair(s)", file=sys.stderr)

    if args.clean:
        samples, clean_rep = clean_dataset(
            samples,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            dedup=args.dedup,
            min_signal_ratio=args.min_signal,
        )
        if not args.quiet:
            print(f"[clean] {clean_rep.as_dict()}", file=sys.stderr)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(quality_report(samples).as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not args.quiet:
            print(f"[report] wrote {report_path}", file=sys.stderr)

    if post_processing:
        if args.output:
            out = Path(args.output)
            if args.val_split > 0:
                train, val = split_dataset(samples, val_fraction=args.val_split, seed=args.seed)
                save_jsonl(train, out.with_suffix(".train.jsonl"))
                save_jsonl(val, out.with_suffix(".val.jsonl"))
                if not args.quiet:
                    print(
                        f"Wrote {len(train)} train / {len(val)} val next to {out}", file=sys.stderr
                    )
            else:
                save_jsonl(samples, out)
                if not args.quiet:
                    print(f"Wrote {len(samples)} sample(s) to {out}", file=sys.stderr)
        else:
            for sample in samples:
                print(json.dumps(sample, ensure_ascii=False))
    elif args.output:
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
