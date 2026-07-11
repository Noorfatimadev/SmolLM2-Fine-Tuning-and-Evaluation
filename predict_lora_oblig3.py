#!/usr/bin/env python3
"""Generate predictions from a LoRA adapter for IN5550 Obligatory 3."""

from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def read_tsv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return pd.read_csv(f, sep="\t")
    return pd.read_csv(path, sep="\t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--base_model", default="/fp/projects01/ec403/hf_models/SmolLM2-1.7B")
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    df = read_tsv(args.input_path)
    prompts = df["prompt"].astype(str).tolist()
    predictions: list[str] = []

    start = time.perf_counter()
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_prompts = [PROMPT_TEMPLATE.format(prompt=p) for p in prompts[i : i + args.batch_size]]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}

        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = encoded["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        predictions.extend([x.strip() for x in decoded])

    elapsed = time.perf_counter() - start
    out_df = pd.DataFrame({"prompt": prompts, "prediction": predictions})
    if "response" in df.columns:
        out_df["response"] = df["response"].astype(str).tolist()
    out_df.to_csv(args.output_path, sep="\t", index=False)

    print(f"Saved predictions to: {args.output_path}")
    print(f"Total generation time seconds: {elapsed:.2f}")
    print(f"Examples: {len(out_df)}")


if __name__ == "__main__":
    main()
