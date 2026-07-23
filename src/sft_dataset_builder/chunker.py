"""Split extracted text/units into chunks according to a chosen strategy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .extractors.base import ExtractedUnit

VALID_STRATEGIES = {"paragraph", "sentence", "fixed", "page", "timestamp"}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u201c(])")


@dataclass
class Chunk:
    """A single chunk of text ready to be turned into a training sample."""

    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    page_number: Optional[int] = None


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _pack_by_size(pieces: list[str], chunk_size: int, overlap: int, sep: str = " ") -> list[str]:
    """Greedily pack a list of text pieces into chunks of ~chunk_size chars,
    with `overlap` characters of trailing context carried into the next chunk.
    """
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = (current + sep + piece).strip() if current else piece
        if len(candidate) <= chunk_size or not current:
            current = candidate
        else:
            chunks.append(current)
            if overlap > 0:
                tail = current[-overlap:]
                current = (tail + sep + piece).strip()
            else:
                current = piece
    if current:
        chunks.append(current)
    return chunks


def chunk_units(
    units: list[ExtractedUnit],
    strategy: str = "paragraph",
    chunk_size: int = 2048,
    overlap: int = 200,
) -> list[Chunk]:
    """Turn a list of ExtractedUnit into a list of Chunk according to strategy.

    - paragraph: each non-empty paragraph/unit is its own chunk (units merged
      up to chunk_size where units are small, split where they're large).
    - sentence: split into sentences, then packed up to chunk_size.
    - fixed: raw fixed-width character windows with overlap, ignoring
      unit/paragraph boundaries.
    - page: one chunk per page_number (falls back to per-unit if no page info).
    - timestamp: one chunk per contiguous group of units sharing timing info
      (audio/video), packed up to chunk_size.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown split_strategy '{strategy}'. Must be one of {sorted(VALID_STRATEGIES)}"
        )

    non_empty_units = [u for u in units if u.text and u.text.strip()]
    if not non_empty_units:
        return []

    if strategy == "paragraph":
        return _chunk_paragraph(non_empty_units, chunk_size, overlap)
    if strategy == "sentence":
        return _chunk_sentence(non_empty_units, chunk_size, overlap)
    if strategy == "fixed":
        return _chunk_fixed(non_empty_units, chunk_size, overlap)
    if strategy == "page":
        return _chunk_page(non_empty_units)
    if strategy == "timestamp":
        return _chunk_timestamp(non_empty_units, chunk_size)

    raise AssertionError("unreachable")  # pragma: no cover


def _chunk_paragraph(units: list[ExtractedUnit], chunk_size: int, overlap: int) -> list[Chunk]:
    texts = [u.text.strip() for u in units]
    packed = _pack_by_size(texts, chunk_size, overlap, sep="\n\n")
    return [Chunk(text=t, index=i) for i, t in enumerate(packed)]


def _chunk_sentence(units: list[ExtractedUnit], chunk_size: int, overlap: int) -> list[Chunk]:
    sentences: list[str] = []
    for u in units:
        sentences.extend(_split_sentences(u.text))
    packed = _pack_by_size(sentences, chunk_size, overlap, sep=" ")
    return [Chunk(text=t, index=i) for i, t in enumerate(packed)]


def _chunk_fixed(units: list[ExtractedUnit], chunk_size: int, overlap: int) -> list[Chunk]:
    full_text = "\n\n".join(u.text.strip() for u in units)
    chunks: list[Chunk] = []
    step = max(chunk_size - overlap, 1)
    idx = 0
    pos = 0
    n = len(full_text)
    while pos < n:
        window = full_text[pos : pos + chunk_size]
        if window.strip():
            chunks.append(Chunk(text=window.strip(), index=idx))
            idx += 1
        pos += step
    return chunks


def _chunk_page(units: list[ExtractedUnit]) -> list[Chunk]:
    pages: dict[Optional[int], list[str]] = {}
    order: list[Optional[int]] = []
    for u in units:
        key = u.page_number
        if key not in pages:
            pages[key] = []
            order.append(key)
        pages[key].append(u.text.strip())

    chunks: list[Chunk] = []
    for i, key in enumerate(order):
        text = "\n\n".join(pages[key])
        chunks.append(Chunk(text=text, index=i, page_number=key))
    return chunks


def _chunk_timestamp(units: list[ExtractedUnit], chunk_size: int) -> list[Chunk]:
    # Group consecutive units (already time-ordered) until chunk_size is reached,
    # keeping the start/end time span of the group.
    chunks: list[Chunk] = []
    buf_texts: list[str] = []
    buf_start: Optional[float] = None
    buf_end: Optional[float] = None
    idx = 0

    def flush() -> None:
        nonlocal buf_texts, buf_start, buf_end, idx
        if buf_texts:
            chunks.append(
                Chunk(
                    text=" ".join(buf_texts).strip(),
                    index=idx,
                    start_time=buf_start,
                    end_time=buf_end,
                )
            )
            idx += 1
        buf_texts = []
        buf_start = None
        buf_end = None

    for u in units:
        candidate_len = len(" ".join([*buf_texts, u.text]))
        if buf_texts and candidate_len > chunk_size:
            flush()
        if buf_start is None:
            buf_start = u.start_time
        buf_end = u.end_time if u.end_time is not None else buf_end
        buf_texts.append(u.text.strip())

    flush()

    # Fallback: if none of the units carried timing info, behave like paragraph chunking.
    if (
        all(c.start_time is None and c.end_time is None for c in chunks)
        and units[0].start_time is None
    ):
        return _chunk_paragraph(units, chunk_size, overlap=0)

    return chunks
