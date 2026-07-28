"""SFT Dataset Builder.

Convert unstructured data (documents, images, audio, video) into structured
SFT training datasets, entirely locally (no cloud APIs).

Quick start
-----------
>>> from sft_dataset_builder import build_dataset
>>> result = build_dataset(
...     files=["doc.pdf", "img.png"],
...     output_format="alpaca",
...     split_strategy="paragraph",
...     save_path="output.jsonl",
... )
>>> result.num_samples
"""

from .capabilities import FormatStatus, probe_capabilities
from .chunker import Chunk, chunk_units
from .curation import (
    CleanReport,
    QualityReport,
    clean_dataset,
    quality_report,
    sample_content,
    split_dataset,
)
from .extractors import SUPPORTED_EXTENSIONS, get_extractor_for
from .formatter import VALID_FORMATS, format_chunk, format_chunks
from .instruction_gen import generate_instruction_samples, openai_compatible_client
from .pipeline import BuildResult, FileReport, build_dataset, save_jsonl

__version__ = "0.3.0"

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "VALID_FORMATS",
    "BuildResult",
    "Chunk",
    "CleanReport",
    "FileReport",
    "FormatStatus",
    "QualityReport",
    "__version__",
    "build_dataset",
    "chunk_units",
    "clean_dataset",
    "format_chunk",
    "format_chunks",
    "generate_instruction_samples",
    "get_extractor_for",
    "openai_compatible_client",
    "probe_capabilities",
    "quality_report",
    "sample_content",
    "save_jsonl",
    "split_dataset",
]
