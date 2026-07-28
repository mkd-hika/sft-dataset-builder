"""Turn extracted chunks into real instruction/response pairs using an LLM.

The base pipeline pairs a constant dummy instruction with each chunk, which is
fine for continued pretraining but not for instruction tuning. This module uses
a language model to synthesize a genuine instruction (question/task) whose answer
is grounded in the chunk, producing proper instruction-tuning samples.

The LLM is pluggable: pass any ``complete(prompt) -> str`` callable. A built-in
:func:`openai_compatible_client` targets an OpenAI-compatible ``/chat/completions``
endpoint (local Ollama / vLLM by default, or any hosted provider) using only the
standard library.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Any, Callable, Optional

from .chunker import Chunk
from .curation import Sample, sample_content
from .formatter import format_chunk

logger = logging.getLogger("sft_dataset_builder")

Completion = Callable[[str], str]

DEFAULT_SYSTEM = (
    "You are a meticulous dataset creator. Given a passage, you write ONE clear, "
    "self-contained instruction (a question or task) whose answer is fully supported "
    "by the passage, plus the answer. Never invent facts beyond the passage."
)

_PROMPT_TEMPLATE = (
    "From the passage below, create ONE instruction/answer pair for supervised "
    "fine-tuning. The instruction must be answerable using only the passage, and the "
    "answer must be grounded in it. Respond with STRICT JSON and nothing else:\n"
    '{{"instruction": "...", "output": "..."}}\n\n'
    'Passage:\n"""\n{content}\n"""'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_prompt(content: str) -> str:
    """Build the instruction-generation prompt for a chunk of text."""
    return _PROMPT_TEMPLATE.format(content=content.strip())


def parse_pair(raw: str) -> Optional[tuple[str, str]]:
    """Parse an ``{"instruction", "output"}`` pair from an LLM response, or None."""
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    instruction = str(obj.get("instruction", "")).strip()
    output = str(obj.get("output", obj.get("answer", ""))).strip()
    if instruction and output:
        return instruction, output
    return None


def generate_instruction_samples(
    samples: list[Sample],
    complete: Completion,
    *,
    output_format: str = "alpaca",
    system_prompt: Optional[str] = None,
    max_items: Optional[int] = None,
    min_chars: int = 64,
    on_error: Optional[Callable[[int, Exception], None]] = None,
) -> list[Sample]:
    """Generate instruction-tuning samples from extracted chunks via ``complete``.

    For each input sample, the LLM is asked to produce an instruction whose answer
    is the passage; the pair is emitted in ``output_format``. Samples whose content
    is shorter than ``min_chars`` are skipped, as are ones the model fails to answer
    or returns unparseable JSON for (reported via ``on_error`` if given).
    """
    generated: list[Sample] = []
    for index, sample in enumerate(samples):
        if max_items is not None and len(generated) >= max_items:
            break
        content = sample_content(sample).strip()
        if len(content) < min_chars:
            continue
        try:
            raw = complete(build_prompt(content))
        except Exception as exc:  # noqa: BLE001 - per-item robustness; keep going
            if on_error is not None:
                on_error(index, exc)
            continue
        pair = parse_pair(raw)
        if pair is None:
            if on_error is not None:
                on_error(index, ValueError("could not parse an instruction/output pair"))
            continue
        instruction, answer = pair
        source_file = sample.get("metadata", {}).get("source_file")
        new_sample = format_chunk(
            Chunk(text=answer, index=index),
            output_format=output_format,
            system_prompt=system_prompt,
            source_file=source_file,
            instruction=instruction,
        )
        new_sample.setdefault("metadata", {})["generated"] = True
        generated.append(new_sample)
    return generated


def openai_compatible_client(
    *,
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen2.5",
    api_key: str = "",
    system: Optional[str] = DEFAULT_SYSTEM,
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> Completion:
    """Return a ``complete(prompt) -> str`` backed by an OpenAI-compatible endpoint.

    Defaults to a local Ollama server (``http://localhost:11434/v1``); works with
    vLLM, LM Studio, OpenAI, or any compatible ``/chat/completions`` API.
    """
    url = base_url.rstrip("/") + "/chat/completions"

    def complete(prompt: str) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body: Any = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])

    return complete
