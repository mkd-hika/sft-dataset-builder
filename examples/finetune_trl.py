"""Fine-tune a small HuggingFace model (LoRA) on a JSONL dataset from this tool.

This is a minimal, illustrative example. The TRL API shifts between versions, so
treat it as a starting point and adjust for your installed versions.

Install (separately from this project):
    pip install "trl>=0.9" "transformers>=4.40" "datasets>=2.18" "peft>=0.10" accelerate

Usage:
    python examples/finetune_trl.py --data dataset.train.jsonl --val dataset.val.jsonl \
        --model Qwen/Qwen2.5-0.5B-Instruct --output sft-out
"""

from __future__ import annotations

import argparse


def to_text(example: dict) -> dict:
    """Normalize any of this tool's output shapes to a single ``text`` field."""
    if "messages" in example:  # openai chat: leave for the chat template
        return example
    if "text" in example:  # raw
        return {"text": example["text"]}
    # alpaca {instruction, input, output} -> a simple prompt/response string
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", example.get("chosen", ""))
    prompt = instruction + (f"\n\n{inp}" if inp else "")
    return {"text": f"### Instruction:\n{prompt}\n\n### Response:\n{output}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="dataset.train.jsonl")
    ap.add_argument("--val", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--output", default="sft-out")
    ap.add_argument("--epochs", type=float, default=1.0)
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    data_files = {"train": args.data}
    if args.val:
        data_files["validation"] = args.val
    ds = load_dataset("json", data_files=data_files)

    train_ds = ds["train"].map(to_text)
    eval_ds = ds["validation"].map(to_text) if "validation" in ds else None

    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        dataset_text_field="text",
        max_seq_length=1024,
    )
    trainer = SFTTrainer(
        model=args.model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=config,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output)
    print("saved LoRA adapter to", args.output)


if __name__ == "__main__":
    main()
