"""Run from the repository root: python examples/quickstart.py."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from sft_dataset_builder import build_dataset  # noqa: E402

output = ROOT / "examples" / "output.jsonl"
result = build_dataset(
    files=["examples/sample.txt"],
    output_format="alpaca",
    split_strategy="paragraph",
    chunk_size=120,
    overlap=0,
    instruction="Read the following reference passage.",
    save_path=output,
)
if not result.num_successful_files or not result.num_samples:
    raise RuntimeError("The example did not produce samples")
print(f"Processed files: {result.num_successful_files}")
print(f"Training samples: {result.num_samples}")
print(json.dumps(result.samples[0], indent=2))
