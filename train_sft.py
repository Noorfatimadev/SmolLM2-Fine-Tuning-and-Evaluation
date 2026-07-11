"""
train_sft.py — Full fine-tuning of SmolLM2-135M / SmolLM2-135M-Instruct
using SFTTrainer from the trl library.

Usage:
    python train_sft.py \
        --model_name HuggingFaceTB/SmolLM2-135M \
        --train_file obli3_train.tsv.gz \
        --output_dir ./output_base_135M \
        --num_train_epochs 2 \
        --learning_rate 2e-5 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --max_seq_length 1024
"""

import os

CACHE_DIR = "/fp/projects01/ec403/hf_models"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

import pandas as pd
from argparse import ArgumentParser
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model ID (e.g. HuggingFaceTB/SmolLM2-135M)")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to obli3_train.tsv.gz")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def format_to_chat(example):
    """
    Converts a (prompt, response) pair into a list of chat messages.
    SFTTrainer expects a 'messages' column with this format.
    """
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return {"messages": messages}


def main():
    args = parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=CACHE_DIR
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set a chat template with {% generation %} markers so that assistant_only_loss can identify which tokens are the assistant response.
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|im_start|>assistant\n"
        "{% generation %}{{ message['content'] }}{% endgeneration %}"
        "<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=CACHE_DIR,
    )

    # Load and prepare data
    print("Loading training data...", flush=True)
    df = pd.read_csv(args.train_file, sep="\t")
    df = df.dropna(subset=["prompt", "response"])
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()
    df = df[(df["prompt"] != "") & (df["response"] != "")]
    print(f"Training examples after cleaning: {len(df)}", flush=True)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_to_chat, remove_columns=["prompt", "response"])

    # Print an example to verify formatting
    test_formatted = tokenizer.apply_chat_template(
        dataset[0]["messages"], tokenize=False, add_generation_prompt=False
    )
    print(f"Example formatted chat:\n{test_formatted}\n", flush=True)

    # Configure training
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        max_length=args.max_seq_length,
        bf16=args.bf16,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=False,
        assistant_only_loss=True,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...", flush=True)
    trainer.train()

    # Save
    print(f"Saving model to {args.output_dir}", flush=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()