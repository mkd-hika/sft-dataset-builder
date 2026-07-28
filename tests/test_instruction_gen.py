"""Tests for LLM-backed instruction generation."""

from __future__ import annotations

import json

import sft_dataset_builder.instruction_gen as ig
from sft_dataset_builder.instruction_gen import (
    build_prompt,
    generate_instruction_samples,
    openai_compatible_client,
    parse_pair,
)


def test_build_prompt_includes_content():
    assert "the passage" in build_prompt("the passage content").lower()


def test_parse_pair_variants():
    assert parse_pair('{"instruction": "Q?", "output": "A."}') == ("Q?", "A.")
    # tolerates surrounding prose
    assert parse_pair('Sure!\n{"instruction": "Q", "output": "A"}\nDone') == ("Q", "A")
    # accepts "answer" as an alias
    assert parse_pair('{"instruction": "Q", "answer": "A"}') == ("Q", "A")
    assert parse_pair("not json") is None
    assert parse_pair('{"instruction": "", "output": "A"}') is None


def test_generate_instruction_samples_happy_path():
    samples = [
        {"text": "x" * 100, "metadata": {"source_file": "a.txt"}},
        {"text": "too short"},  # below min_chars -> skipped
    ]

    def fake_complete(prompt: str) -> str:
        return '{"instruction": "Summarize this.", "output": "A summary."}'

    out = generate_instruction_samples(samples, fake_complete, output_format="alpaca", min_chars=64)
    assert len(out) == 1
    assert out[0]["instruction"] == "Summarize this."
    assert out[0]["output"] == "A summary."
    assert out[0]["metadata"]["generated"] is True
    assert out[0]["metadata"]["source_file"] == "a.txt"


def test_generate_reports_errors_and_continues():
    samples = [{"text": "y" * 100}, {"text": "z" * 100}]
    errors = []

    def flaky(prompt: str) -> str:
        return "not parseable json"

    out = generate_instruction_samples(
        samples, flaky, min_chars=10, on_error=lambda i, exc: errors.append(i)
    )
    assert out == []
    assert len(errors) == 2


def test_max_items_caps_output():
    samples = [{"text": "w" * 100} for _ in range(5)]
    out = generate_instruction_samples(
        samples, lambda p: '{"instruction":"Q","output":"A"}', min_chars=10, max_items=2
    )
    assert len(out) == 2


def test_openai_compatible_client_builds_and_parses(monkeypatch):
    captured = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "HELLO"}}]}).encode()

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        return FakeResp()

    monkeypatch.setattr(ig.urllib.request, "urlopen", fake_urlopen)
    complete = openai_compatible_client(base_url="http://x/v1", model="m", api_key="k")
    assert complete("hi there") == "HELLO"
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["messages"][-1]["content"] == "hi there"
