#!/usr/bin/env python3
"""LoRA fine-tuning for IN5550 Obligatory 3, section 3.3."""

from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


PROMPT_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def read_tsv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return pd.read_csv(f, sep="\t")
    return pd.read_csv(path, sep="\t")


class InstructionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.rows = df[["prompt", "response"]].to_dict("records")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        row = self.rows[idx]
        prefix = PROMPT_TEMPLATE.format(prompt=row["prompt"])
        full_text = prefix + str(row["response"]) + self.tokenizer.eos_token

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        encoded = self.tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=self.max_length)

        input_ids = encoded["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prefix_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": input_ids, "attention_mask": encoded["attention_mask"], "labels": labels}


@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--model_name_or_path", default="/fp/projects01/ec403/hf_models/SmolLM2-1.7B")
    parser.add_argument("--output_dir", type=Path, default=Path("results_lora/smollm2_1_7b_lora"))
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = read_tsv(args.train_path)
    if args.max_train_examples:
        df = df.sample(n=min(args.max_train_examples, len(df)), random_state=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = InstructionDataset(df, tokenizer, args.max_length)

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=20,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForCausalLM(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))

    with open(args.output_dir / "lora_config_used.txt", "w", encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    print("Saved LoRA adapter to:", args.output_dir / "adapter")


if __name__ == "__main__":
    main()
