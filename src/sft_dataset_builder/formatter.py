"""Format chunks into SFT training sample dicts (ready to write as JSONL)."""

from __future__ import annotations

from typing import Any, Optional

from .chunker import Chunk

VALID_FORMATS = {"alpaca", "sharegpt", "openai", "raw", "dpo", "orpo"}

DEFAULT_INSTRUCTION = "Read and understand the following content."


def format_chunk(
    chunk: Chunk,
    output_format: str,
    system_prompt: Optional[str] = None,
    source_file: Optional[str] = None,
    instruction: Optional[str] = None,
) -> dict[str, Any]:
    """Convert a single Chunk into a training-sample dict in `output_format`."""
    if output_format not in VALID_FORMATS:
        raise ValueError(
            f"Unknown output_format '{output_format}'. Must be one of {sorted(VALID_FORMATS)}"
        )

    text = chunk.text.strip()
    instr = instruction or DEFAULT_INSTRUCTION
    sys_prompt = system_prompt or "You are a helpful assistant."

    sample: dict[str, Any]
    if output_format == "alpaca":
        sample = {
            "instruction": instr,
            "input": "",
            "output": text,
        }
    elif output_format == "sharegpt":
        conversations = []
        if system_prompt:
            conversations.append({"from": "system", "value": sys_prompt})
        conversations.append({"from": "human", "value": instr})
        conversations.append({"from": "gpt", "value": text})
        sample = {"conversations": conversations}
    elif output_format == "openai":
        messages = [{"role": "system", "content": sys_prompt}]
        messages.append({"role": "user", "content": instr})
        messages.append({"role": "assistant", "content": text})
        sample = {"messages": messages}
    elif output_format == "raw":
        sample = {"text": text}
    elif output_format == "dpo":
        # Placeholder preference-pair scaffold: `chosen` is the extracted text,
        # `rejected` is left blank for the user to fill in (e.g. via a weaker
        # model or manual curation) since we have no negative sample source.
        sample = {
            "prompt": instr,
            "chosen": text,
            "rejected": "",
        }
    elif output_format == "orpo":
        sample = {
            "prompt": instr,
            "chosen": text,
            "rejected": "",
        }
    else:  # pragma: no cover - guarded above
        raise AssertionError("unreachable")

    meta: dict[str, Any] = {"chunk_index": chunk.index}
    if source_file:
        meta["source_file"] = source_file
    if chunk.page_number is not None:
        meta["page_number"] = chunk.page_number
    if chunk.start_time is not None:
        meta["start_time"] = chunk.start_time
    if chunk.end_time is not None:
        meta["end_time"] = chunk.end_time
    sample["metadata"] = meta

    return sample


def format_chunks(
    chunks: list[Chunk],
    output_format: str,
    system_prompt: Optional[str] = None,
    source_file: Optional[str] = None,
    instruction: Optional[str] = None,
) -> list[dict[str, Any]]:
    return [
        format_chunk(
            chunk,
            output_format=output_format,
            system_prompt=system_prompt,
            source_file=source_file,
            instruction=instruction,
        )
        for chunk in chunks
    ]
