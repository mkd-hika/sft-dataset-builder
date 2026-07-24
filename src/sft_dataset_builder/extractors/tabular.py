"""Extractor for delimited tabular data (.csv/.tsv).

Default mapping: the first row is treated as a header, and each subsequent row
becomes one unit rendered as ``column: value`` lines. This keeps column context
with every sample. If no header/columns can be inferred, rows are emitted as
pipe-joined values instead.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, ClassVar

from .base import BaseExtractor, ExtractedUnit, ExtractionResult
from .text import _read_text_file


class TabularExtractor(BaseExtractor):
    """Handles .csv and .tsv, one row per unit (header-aware)."""

    supported_extensions: ClassVar[set[str]] = {".csv", ".tsv"}

    def extract(self, path: Path, **kwargs: Any) -> ExtractionResult:
        raw = _read_text_file(path)
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        source_type = path.suffix.lower().lstrip(".")

        rows = [
            row
            for row in csv.reader(raw.splitlines(), delimiter=delimiter)
            if any(c.strip() for c in row)
        ]
        if not rows:
            return ExtractionResult(
                source_path=str(path),
                source_type=source_type,
                units=[],
                warnings=["no rows found"],
            )

        header = [c.strip() for c in rows[0]]
        units: list[ExtractedUnit] = []
        for i, row in enumerate(rows[1:]):
            pairs = [f"{col}: {val.strip()}" for col, val in zip(header, row) if val.strip()]
            if pairs:
                units.append(ExtractedUnit(text="\n".join(pairs), metadata={"row_index": i}))

        if not units:  # single-column or headerless data — emit raw rows
            units = [
                ExtractedUnit(text=" | ".join(c.strip() for c in row), metadata={"row_index": i})
                for i, row in enumerate(rows)
            ]

        return ExtractionResult(source_path=str(path), source_type=source_type, units=units)
