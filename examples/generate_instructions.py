"""Extract -> generate REAL instruction pairs (local LLM) -> clean -> split.

This turns the builder's raw chunks into genuine instruction-tuning data using a
local, OpenAI-compatible LLM endpoint, then cleans and splits the result.

Prerequisites (local, offline):
    ollama serve            # or any OpenAI-compatible server (vLLM, LM Studio, ...)
    ollama pull qwen2.5

Usage:
    python examples/generate_instructions.py corpus.pdf notes.md -o dataset.jsonl \
        --model qwen2.5 --max-items 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sft_dataset_builder import (
    build_dataset,
    clean_dataset,
    generate_instruction_samples,
    openai_compatible_client,
    save_jsonl,
    split_dataset,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="Source files to extract from.")
    ap.add_argument("-o", "--output", default="instruct.jsonl", help="Output stem for JSONL.")
    ap.add_argument("--base-url", default="http://localhost:11434/v1", help="LLM endpoint.")
    ap.add_argument("--model", default="qwen2.5", help="Model name on the endpoint.")
    ap.add_argument("--api-key", default="", help="API key, if the endpoint needs one.")
    ap.add_argument("--max-items", type=int, default=None, help="Cap generated pairs.")
    ap.add_argument("--val-split", type=float, default=0.1, help="Validation fraction.")
    args = ap.parse_args()

    built = build_dataset(args.files, output_format="raw", split_strategy="paragraph")
    print(f"extracted {built.num_samples} chunks from {built.num_successful_files} file(s)")

    complete = openai_compatible_client(
        base_url=args.base_url, model=args.model, api_key=args.api_key
    )
    samples = generate_instruction_samples(
        built.samples,
        complete,
        output_format="alpaca",
        max_items=args.max_items,
        on_error=lambda i, exc: print(f"  chunk {i}: {exc}"),
    )
    print(f"generated {len(samples)} instruction/answer pairs")

    kept, report = clean_dataset(samples)
    print("cleaning:", report.as_dict())

    train, val = split_dataset(kept, val_fraction=args.val_split)
    out = Path(args.output)
    save_jsonl(train, out.with_suffix(".train.jsonl"))
    save_jsonl(val, out.with_suffix(".val.jsonl"))
    print(f"wrote {len(train)} train / {len(val)} val next to {out}")


if __name__ == "__main__":
    main()
