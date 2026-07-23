"""Tests for the end-to-end pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from sft_dataset_builder.pipeline import BuildResult, build_dataset, save_jsonl


def test_build_from_text_file(txt_file: Path):
    result = build_dataset([txt_file], output_format="alpaca", split_strategy="paragraph")
    assert isinstance(result, BuildResult)
    assert result.num_files == 1
    assert result.num_successful_files == 1
    assert result.num_samples >= 1
    assert result.total_chars > 0
    assert result.estimated_size_bytes() > 0
    report = result.file_reports[0]
    assert report.success
    assert report.num_samples == result.num_samples


def test_missing_file_is_reported_not_raised(tmp_path: Path):
    result = build_dataset([tmp_path / "does_not_exist.txt"])
    assert result.num_files == 1
    assert result.num_successful_files == 0
    assert result.file_reports[0].error == "File not found"


def test_unsupported_type_is_reported(tmp_path: Path):
    weird = tmp_path / "thing.xyz"
    weird.write_text("data", encoding="utf-8")
    result = build_dataset([weird])
    assert result.num_successful_files == 0
    assert "Unsupported file type" in result.file_reports[0].error


def test_mixed_batch_continues_after_failure(txt_file: Path, tmp_path: Path):
    missing = tmp_path / "nope.txt"
    result = build_dataset([missing, txt_file])
    assert result.num_files == 2
    assert result.num_successful_files == 1
    assert result.num_samples >= 1


def test_save_path_writes_valid_jsonl(txt_file: Path, tmp_path: Path):
    out = tmp_path / "nested" / "out.jsonl"
    result = build_dataset([txt_file], save_path=out)
    assert out.exists()
    assert result.save_path == str(out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == result.num_samples
    for line in lines:
        json.loads(line)  # each line is valid JSON


def test_save_jsonl_roundtrip(tmp_path: Path):
    samples = [{"text": "a", "metadata": {"i": 0}}, {"text": "b", "metadata": {"i": 1}}]
    out = tmp_path / "data.jsonl"
    save_jsonl(samples, out)
    loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert loaded == samples


def test_progress_callback_receives_stages(txt_file: Path):
    stages: list[str] = []
    build_dataset([txt_file], progress_callback=lambda s, m, e: stages.append(s))
    assert "start" in stages
    assert "done" in stages


def test_progress_callback_errors_do_not_break_pipeline(txt_file: Path):
    def boom(stage, message, extra):
        raise RuntimeError("callback blew up")

    # Should not raise despite the misbehaving callback.
    result = build_dataset([txt_file], progress_callback=boom)
    assert result.num_successful_files == 1
