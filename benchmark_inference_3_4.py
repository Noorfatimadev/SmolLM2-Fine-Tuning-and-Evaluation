#!/usr/bin/env python3
"""Inference speed comparison for IN5550 Obligatory 3, section 3.4."""

from __future__ import annotations
 
import argparse
import gzip
import time
from pathlib import Path

import pandas as pd


PROMPT_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def read_tsv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return pd.read_csv(f, sep="\t")
    return pd.read_csv(path, sep="\t")


def write_outputs(path: Path, prompts: list[str], predictions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prompt": prompts, "prediction": predictions}).to_csv(path, sep="\t", index=False)


def run_hf(args: argparse.Namespace, prompts: list[str]) -> tuple[list[str], float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    predictions: list[str] = []
    start = time.perf_counter()
    for i in range(0, len(prompts), args.batch_size):
        batch = [PROMPT_TEMPLATE.format(prompt=p) for p in prompts[i : i + args.batch_size]]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_tokens)
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        predictions.extend(tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True))
    elapsed = time.perf_counter() - start
    return [p.strip() for p in predictions], elapsed


def run_vllm(args: argparse.Namespace, prompts: list[str]) -> tuple[list[str], float]:
    from vllm import LLM, SamplingParams

    formatted = [PROMPT_TEMPLATE.format(prompt=p) for p in prompts]
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    llm = LLM(
        model=args.model_name_or_path,
        max_model_len=args.max_input_tokens + args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    start = time.perf_counter()
    outputs = llm.generate(formatted, sampling_params)
    elapsed = time.perf_counter() - start
    predictions = [out.outputs[0].text.strip() if out.outputs else "" for out in outputs]
    return predictions, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("results_inference_speed"))
    parser.add_argument("--backend", choices=["hf", "vllm"], required=True)
    parser.add_argument("--model_name_or_path", default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = read_tsv(args.input_path)
    if args.max_examples:
        df = df.head(args.max_examples)
    prompts = df["prompt"].astype(str).tolist()

    if args.backend == "hf":
        predictions, elapsed = run_hf(args, prompts)
    else:
        predictions, elapsed = run_vllm(args, prompts)

    output_path = args.out_dir / f"{args.backend}_predictions.tsv"
    metrics_path = args.out_dir / f"{args.backend}_speed.csv"
    write_outputs(output_path, prompts, predictions)

    metrics = pd.DataFrame(
        [
            {
                "backend": args.backend,
                "model": args.model_name_or_path,
                "n_examples": len(prompts),
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size if args.backend == "hf" else "vllm_dynamic",
                "total_seconds": elapsed,
                "examples_per_second": len(prompts) / elapsed if elapsed > 0 else 0,
            }
        ]
    )
    metrics.to_csv(metrics_path, index=False)
    print(metrics.to_string(index=False))
    print("Saved predictions to:", output_path)
    print("Saved speed metrics to:", metrics_path)


if __name__ == "__main__":
    main()
