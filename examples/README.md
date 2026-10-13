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

## Additional examples

The existing generation and training examples below require their respective model dependencies; they were not executed in this presentation update.

## `generate_instructions.py` — build real instruction-tuning data

The base pipeline pairs a constant instruction with each extracted chunk (great
for continued pretraining, not for instruction tuning). This example uses a
**local LLM** to synthesize a genuine instruction whose answer is the chunk,
then cleans and splits the result.

```bash
# 1) start a local, OpenAI-compatible LLM (offline)
ollama serve
ollama pull qwen2.5

# 2) extract -> generate instruction pairs -> clean -> split
python examples/generate_instructions.py corpus.pdf notes.md \
    -o dataset.jsonl --model qwen2.5 --max-items 200
# -> dataset.train.jsonl, dataset.val.jsonl
```

Works with any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, or a hosted
provider) via `--base-url` / `--model` / `--api-key`.

## `finetune_trl.py` — fine-tune a small model (LoRA)

Loads the JSONL and fine-tunes a small HuggingFace model with TRL's `SFTTrainer`.
Illustrative — adjust for your installed TRL/transformers versions.

```bash
pip install "trl>=0.9" "transformers>=4.40" "datasets>=2.18" "peft>=0.10" accelerate

python examples/finetune_trl.py \
    --data dataset.train.jsonl --val dataset.val.jsonl \
    --model Qwen/Qwen2.5-0.5B-Instruct --output sft-out
```

## Which output format for which trainer?

| Format | Use with |
|--------|----------|
| `raw` (`{"text"}`) | TRL `SFTTrainer` for continued pretraining / plain SFT |
| `openai` (`{"messages"}`) | TRL conversational SFT (applies the chat template) |
| `alpaca` / `sharegpt` | axolotl, LLaMA-Factory (native), or map to `text` (see `to_text`) |
| `dpo` / `orpo` | preference training — **fill in `rejected` first** |
