# Runnable text example

From a fresh clone, run this from the repository root:

```bash
python examples/quickstart.py
```

The text-only example uses the repository's source directly and requires no third-party packages, network access, or GPU. It processes [sample.txt](sample.txt), writes [output.jsonl](output.jsonl), and prints the first sample. The supplied passages are synthetic.

## Verified output

One source file produces **3 Alpaca-format samples** with `chunk_size=120` and no overlap. The first row is:

```json
{
  "instruction": "Read the following reference passage.",
  "input": "",
  "output": "Dataset preparation starts by extracting readable text from source files.",
  "metadata": {
    "chunk_index": 0,
    "source_file": "examples\\sample.txt"
  }
}
```

The instruction is a reading prompt paired with source text. These are extracted passages, not automatically generated factual question/answer pairs. Review and curate them for your training objective.

## Open the UI

```bash
python -m pip install -e ".[ui]"
sft-dataset-builder-ui
```

Upload `examples/sample.txt` to explore the same source through the interface. Other formats need their corresponding optional dependencies.
