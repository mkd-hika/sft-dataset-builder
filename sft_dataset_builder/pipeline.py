"""Orchestrates the full extract -> chunk -> format -> save pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .chunker import chunk_units
from .extractors import get_extractor_for
from .formatter import format_chunks

logger = logging.getLogger("sft_dataset_builder")

ProgressCallback = Callable[[str, str, dict[str, Any]], None]
# callback(stage, message, extra) -> None
# stage in {"start", "extracting", "chunking", "formatting", "done", "error", "warning"}


@dataclass
class FileReport:
    path: str
    success: bool
    char_count: int = 0
    num_units: int = 0
    num_chunks: int = 0
    num_samples: int = 0
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class BuildResult:
    samples: list[dict[str, Any]] = field(default_factory=list)
    file_reports: list[FileReport] = field(default_factory=list)
    save_path: Optional[str] = None

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def num_files(self) -> int:
        return len(self.file_reports)

    @property
    def num_successful_files(self) -> int:
        return sum(1 for r in self.file_reports if r.success)

    @property
    def total_chars(self) -> int:
        return sum(r.char_count for r in self.file_reports)

    def estimated_size_bytes(self) -> int:
        return sum(len(json.dumps(s, ensure_ascii=False)) + 1 for s in self.samples)


def build_dataset(
    files: list[Union[str, Path]],
    output_format: str = "alpaca",
    split_strategy: str = "paragraph",
    chunk_size: int = 2048,
    overlap: int = 200,
    system_prompt: Optional[str] = "You are a helpful assistant.",
    instruction: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    extractor_kwargs: Optional[dict[str, Any]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> BuildResult:
    """Run the full pipeline over `files` and return a BuildResult.

    Parameters
    ----------
    files: paths to source files (any mix of supported types).
    output_format: one of alpaca, sharegpt, openai, raw, dpo, orpo.
    split_strategy: one of paragraph, sentence, fixed, page, timestamp.
    chunk_size: target max characters per chunk.
    overlap: character overlap between consecutive chunks (paragraph/sentence/fixed).
    system_prompt: system prompt injected into sharegpt/openai formats.
    instruction: instruction text used for alpaca/sharegpt/openai/dpo/orpo prompts.
        Defaults to a generic instruction if not provided.
    save_path: if given, writes the resulting samples to this path as JSONL.
    extractor_kwargs: optional dict of extra kwargs passed to every extractor's
        `.extract()` call (e.g. {"model_size": "small", "ocr_images": True}).
    progress_callback: optional callable(stage, message, extra) for progress UIs.
    """
    extractor_kwargs = extractor_kwargs or {}
    result = BuildResult()

    def notify(stage: str, message: str, **extra: Any) -> None:
        if progress_callback:
            try:
                progress_callback(stage, message, extra)
            except Exception:
                # Never let a UI/progress callback break the pipeline.
                logger.exception("progress_callback raised")

    total = len(files)
    for i, raw_path in enumerate(files, start=1):
        path = Path(raw_path)
        notify("start", f"[{i}/{total}] Processing {path.name}", file=str(path))

        if not path.exists():
            report = FileReport(path=str(path), success=False, error="File not found")
            result.file_reports.append(report)
            notify("error", f"{path.name}: file not found", file=str(path))
            continue

        extractor = get_extractor_for(path)
        if extractor is None:
            report = FileReport(
                path=str(path), success=False, error=f"Unsupported file type: {path.suffix}"
            )
            result.file_reports.append(report)
            notify("error", f"{path.name}: unsupported file type", file=str(path))
            continue

        notify("extracting", f"Extracting content from {path.name}", file=str(path))
        extraction = extractor.safe_extract(path, **extractor_kwargs)

        if not extraction.success:
            report = FileReport(path=str(path), success=False, error=extraction.error)
            result.file_reports.append(report)
            notify("error", f"{path.name}: {extraction.error}", file=str(path))
            continue

        for w in extraction.warnings:
            notify("warning", f"{path.name}: {w}", file=str(path))

        notify(
            "chunking",
            f"Chunking {path.name} ({extraction.char_count} chars, {len(extraction.units)} units)",
            file=str(path),
        )
        try:
            chunks = chunk_units(
                extraction.units,
                strategy=split_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        except Exception as exc:  # noqa: BLE001
            report = FileReport(
                path=str(path),
                success=False,
                char_count=extraction.char_count,
                num_units=len(extraction.units),
                error=f"Chunking failed: {exc}",
                warnings=extraction.warnings,
            )
            result.file_reports.append(report)
            notify("error", f"{path.name}: chunking failed ({exc})", file=str(path))
            continue

        notify("formatting", f"Formatting {len(chunks)} chunks from {path.name}", file=str(path))
        samples = format_chunks(
            chunks,
            output_format=output_format,
            system_prompt=system_prompt,
            source_file=str(path),
            instruction=instruction,
        )
        result.samples.extend(samples)

        report = FileReport(
            path=str(path),
            success=True,
            char_count=extraction.char_count,
            num_units=len(extraction.units),
            num_chunks=len(chunks),
            num_samples=len(samples),
            warnings=extraction.warnings,
        )
        result.file_reports.append(report)
        notify(
            "done",
            f"Done: {path.name} -> {len(samples)} samples",
            file=str(path),
            num_samples=len(samples),
        )

    if save_path:
        save_jsonl(result.samples, save_path)
        result.save_path = str(save_path)
        notify("done", f"Saved {len(result.samples)} samples to {save_path}")

    return result


def save_jsonl(samples: list[dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
