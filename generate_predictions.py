"""
generate_predictions.py — Generate predictions from a fine-tuned model on the test set.

Usage:
    python generate_predictions.py \
        --model_dir ./output_base_135M \
        --test_file obli3_test.tsv.gz \
        --output_file predictions_base_135M.tsv \
        --max_new_tokens 512 \
        --batch_size 16
"""

import os

CACHE_DIR = "/cluster/work/projects/ec403/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

import torch
import pandas as pd
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to fine-tuned model directory")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to obli3_test.tsv.gz")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output TSV file path")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    #  Load model and tokenizer
    print(f"Loading model from {args.model_dir}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    #  Load test data
    print("Loading test data...", flush=True)
    df = pd.read_csv(args.test_file, sep="\t")
    prompts = df["prompt"].tolist()
    print(f"Number of test examples: {len(prompts)}", flush=True)

    # Format prompts using chat template
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(formatted)

    # Generate in batches
    all_predictions = []
    num_batches = (len(formatted_prompts) + args.batch_size - 1) // args.batch_size

    for i in range(0, len(formatted_prompts), args.batch_size):
        batch = formatted_prompts[i : i + args.batch_size]
        batch_idx = i // args.batch_size + 1
        print(f"Generating batch {batch_idx}/{num_batches}...", flush=True)

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,          # greedy for reproducibility
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens (skip the prompt)
        for j, output in enumerate(outputs):
            input_length = inputs["input_ids"][j].shape[0]
            generated_tokens = output[input_length:]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            all_predictions.append(decoded.strip())

    # Save predictions
    result_df = pd.DataFrame({
        "prompt": prompts,
        "prediction": all_predictions,
    })
    result_df.to_csv(args.output_file, sep="\t", index=False)
    print(f"Predictions saved to {args.output_file}", flush=True)
    print(f"Example prediction:\n{all_predictions[0][:300]}", flush=True)


if __name__ == "__main__":
    main()
