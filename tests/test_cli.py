"""Tests for the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sft_dataset_builder import __version__
from sft_dataset_builder.cli import main


def test_writes_output_file_and_returns_zero(txt_file: Path, tmp_path: Path):
    out = tmp_path / "out.jsonl"
    rc = main([str(txt_file), "-o", str(out), "--output-format", "raw"])
    assert rc == 0
    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines and all(json.loads(line) for line in lines)


def test_streams_jsonl_to_stdout(txt_file: Path, capsys):
    rc = main([str(txt_file), "--output-format", "raw", "--quiet"])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out
    first = json.loads(out[0])
    assert "text" in first


def test_all_files_failing_returns_one(tmp_path: Path):
    weird = tmp_path / "thing.xyz"
    weird.write_text("data", encoding="utf-8")
    rc = main([str(weird), "--quiet"])
    assert rc == 1


def test_no_files_is_argument_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--list-formats", "--list-strategies", "--list-extensions"])
def test_list_flags(flag, capsys):
    rc = main([flag])
    assert rc == 0
    assert capsys.readouterr().out.strip()


def test_invalid_format_is_rejected(txt_file: Path):
    with pytest.raises(SystemExit) as exc:
        main([str(txt_file), "--output-format", "bogus"])
    assert exc.value.code == 2
