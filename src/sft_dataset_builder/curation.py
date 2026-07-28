"""Post-processing for built datasets: clean, deduplicate, split, and report.

These operate on the sample dicts produced by :func:`build_dataset` (any output
format) and are pure-Python with no third-party dependencies.

>>> from sft_dataset_builder import build_dataset, clean_dataset, split_dataset
>>> result = build_dataset(["corpus.pdf"], output_format="raw")
>>> kept, report = clean_dataset(result.samples)
>>> train, val = split_dataset(kept, val_fraction=0.1, seed=0)
"""

from __future__ import annotations

import random
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

Sample = dict[str, Any]

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def sample_content(sample: Sample) -> str:
    """Return the extracted-text portion of a sample, regardless of format.

    Handles raw/alpaca/dpo/orpo/openai/sharegpt shapes; falls back to "".
    """
    if "text" in sample:
        return str(sample["text"])
    if "output" in sample:
        return str(sample["output"])
    if "chosen" in sample:
        return str(sample["chosen"])
    messages = sample.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
    conversations = sample.get("conversations")
    if isinstance(conversations, list):
        for turn in reversed(conversations):
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                return str(turn.get("value", ""))
    return ""


def _normalized_key(text: str) -> str:
    """Whitespace/case/punctuation-insensitive key for near-duplicate detection."""
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


def _signal_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are alphanumeric (0..1)."""
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    return sum(c.isalnum() for c in non_ws) / len(non_ws)


@dataclass
class CleanReport:
    """Summary of what :func:`clean_dataset` kept and dropped, and why."""

    input_count: int = 0
    kept_count: int = 0
    dropped_empty: int = 0
    dropped_short: int = 0
    dropped_long: int = 0
    dropped_low_signal: int = 0
    dropped_duplicate: int = 0

    @property
    def dropped_total(self) -> int:
        return self.input_count - self.kept_count

    def as_dict(self) -> dict[str, int]:
        d = asdict(self)
        d["dropped_total"] = self.dropped_total
        return d


def clean_dataset(
    samples: list[Sample],
    *,
    min_chars: int = 32,
    max_chars: Optional[int] = None,
    dedup: bool = True,
    min_signal_ratio: float = 0.2,
) -> tuple[list[Sample], CleanReport]:
    """Filter and deduplicate samples by their extracted content.

    Drops (in order) empty content, content shorter than ``min_chars`` or longer
    than ``max_chars``, low-signal content (mostly punctuation/symbols — typical
    of OCR noise) below ``min_signal_ratio``, and near-duplicates (normalized,
    keeping the first occurrence). Returns the kept samples and a CleanReport.
    """
    report = CleanReport(input_count=len(samples))
    seen: set[str] = set()
    kept: list[Sample] = []

    for sample in samples:
        content = sample_content(sample).strip()
        if not content:
            report.dropped_empty += 1
            continue
        if len(content) < min_chars:
            report.dropped_short += 1
            continue
        if max_chars is not None and len(content) > max_chars:
            report.dropped_long += 1
            continue
        if min_signal_ratio > 0 and _signal_ratio(content) < min_signal_ratio:
            report.dropped_low_signal += 1
            continue
        if dedup:
            key = _normalized_key(content)
            if key in seen:
                report.dropped_duplicate += 1
                continue
            seen.add(key)
        kept.append(sample)

    report.kept_count = len(kept)
    return kept, report


def split_dataset(
    samples: list[Sample],
    val_fraction: float = 0.1,
    *,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into (train, val). Deterministic for a given ``seed``.

    ``val_fraction`` is clamped to [0, 1]; the validation set gets
    ``round(len(samples) * val_fraction)`` items.
    """
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    ordered = list(samples)
    if shuffle:
        random.Random(seed).shuffle(ordered)
    n_val = round(len(ordered) * val_fraction)
    val = ordered[:n_val]
    train = ordered[n_val:]
    return train, val


@dataclass
class QualityReport:
    """Descriptive statistics for a set of samples."""

    count: int = 0
    empty: int = 0
    unique_content: int = 0
    duplicate_content: int = 0
    char_min: int = 0
    char_max: int = 0
    char_mean: float = 0.0
    char_median: float = 0.0
    char_total: int = 0
    by_source: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def quality_report(samples: list[Sample]) -> QualityReport:
    """Compute descriptive statistics over a dataset's content and sources."""
    report = QualityReport(count=len(samples))
    lengths: list[int] = []
    keys: set[str] = set()
    for sample in samples:
        content = sample_content(sample).strip()
        if not content:
            report.empty += 1
        else:
            keys.add(_normalized_key(content))
        lengths.append(len(content))
        source = str(sample.get("metadata", {}).get("source_file", "unknown"))
        report.by_source[source] = report.by_source.get(source, 0) + 1

    if lengths:
        report.char_min = min(lengths)
        report.char_max = max(lengths)
        report.char_mean = round(statistics.fmean(lengths), 1)
        report.char_median = float(statistics.median(lengths))
        report.char_total = sum(lengths)
    non_empty = report.count - report.empty
    report.unique_content = len(keys)
    report.duplicate_content = max(non_empty - len(keys), 0)
    return report
