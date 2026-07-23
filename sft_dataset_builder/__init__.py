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

from .chunker import Chunk, chunk_units
from .extractors import SUPPORTED_EXTENSIONS, get_extractor_for
from .formatter import VALID_FORMATS, format_chunk, format_chunks
from .pipeline import BuildResult, FileReport, build_dataset, save_jsonl

__version__ = "0.2.0"

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "VALID_FORMATS",
    "BuildResult",
    "Chunk",
    "FileReport",
    "__version__",
    "build_dataset",
    "chunk_units",
    "format_chunk",
    "format_chunks",
    "get_extractor_for",
    "save_jsonl",
]
