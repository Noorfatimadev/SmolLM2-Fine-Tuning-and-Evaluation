"""
evaluate_metrics.py — Evaluate predictions with ROUGE-L, BERTScore,
and analyze by response length and prompt type.

Usage:
    python evaluate_metrics.py \
        --prediction_file predictions_base_135M.tsv \
        --gold_file obli3_test.tsv.gz \
        --batch_size 32
"""

import os

CACHE_DIR = "/cluster/work/projects/ec403/ec-mykhailk/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR + "/modules"

import json
import numpy as np
import pandas as pd
import evaluate
from argparse import ArgumentParser
from rouge_score import rouge_scorer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args()


def classify_prompt_type(prompt):
    prompt_lower = prompt.lower().strip()

    # Question-type prompts
    if any(prompt_lower.startswith(w) for w in
           ["what", "who", "where", "when", "why", "how", "is ", "are ", "do ", "does ",
            "can ", "could ", "would ", "should ", "which "]):
        return "question"

    # Reasoning / math / logic
    if any(keyword in prompt_lower for keyword in
           ["calculate", "compute", "solve", "equation", "math", "sum", "average",
            "probability", "if a ", "if an ", "how many", "how much"]):
        return "reasoning"

    # Creative writing
    if any(keyword in prompt_lower for keyword in
           ["write a story", "write a poem", "write a song", "creative",
            "imagine", "fiction", "compose", "draft a letter"]):
        return "creative"

    # Explanation / instruction (default for imperative sentences)
    if any(prompt_lower.startswith(w) for w in
           ["explain", "describe", "list", "summarize", "define", "compare",
            "outline", "provide", "give", "tell", "name", "identify"]):
        return "explanation"

    return "instruction"


def assign_length_bucket(length):
    """Assign response to short/medium/long bucket based on word count."""
    if length < 50:
        return "short (<50 words)"
    elif length < 150:
        return "medium (50-150 words)"
    else:
        return "long (>150 words)"


def main():
    args = parse_args()

    # Load data
    gold = pd.read_csv(args.gold_file, sep="\t")
    preds = pd.read_csv(args.prediction_file, sep="\t")

    merged = gold.merge(preds, on="prompt", how="inner")
    merged = merged.dropna(subset=["response", "prediction"])
    merged["response"] = merged["response"].astype(str).str.strip()
    merged["prediction"] = merged["prediction"].astype(str).str.strip()
    merged = merged[(merged["response"] != "") & (merged["prediction"] != "")]
    print(f"Evaluating {len(merged)} examples", flush=True)

    predictions = merged["prediction"].tolist()
    references = merged["response"].tolist()

    # ROUGE-L
    print("Computing ROUGE-L...", flush=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    rougeL = round(sum(rouge_scores) / len(rouge_scores) * 100, 3)
    print(f"ROUGE-L: {rougeL}", flush=True)

    # BERTScore
    print("Computing BERTScore...", flush=True)
    bertscore_metric = evaluate.load("bertscore", cache_dir=CACHE_DIR)
    bertscore_result = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        lang="en",
        batch_size=args.batch_size,
    )
    bertscore_f1 = round(np.mean(bertscore_result["f1"]) * 100, 3)
    print(f"BERTScore F1: {bertscore_f1}", flush=True)

    # Per-example scores for analysis
    merged["rougeL"] = rouge_scores
    merged["bertscore_f1"] = bertscore_result["f1"]

    #  Analysis by response length
    print("\n=== Analysis by Reference Response Length ===", flush=True)
    merged["ref_word_count"] = merged["response"].apply(lambda x: len(x.split()))
    merged["length_bucket"] = merged["ref_word_count"].apply(assign_length_bucket)

    length_analysis = merged.groupby("length_bucket").agg(
        count=("rougeL", "count"),
        avg_rougeL=("rougeL", "mean"),
        avg_bertscore=("bertscore_f1", "mean"),
    ).round(4)
    print(length_analysis.to_string(), flush=True)

    # Analysis by prompt type
    print("\n=== Analysis by Prompt Type ===", flush=True)
    merged["prompt_type"] = merged["prompt"].apply(classify_prompt_type)

    type_analysis = merged.groupby("prompt_type").agg(
        count=("rougeL", "count"),
        avg_rougeL=("rougeL", "mean"),
        avg_bertscore=("bertscore_f1", "mean"),
    ).round(4)
    print(type_analysis.to_string(), flush=True)

    # Also analyze by prediction length
    print("\n=== Analysis by Prediction Length ===", flush=True)
    merged["pred_word_count"] = merged["prediction"].apply(lambda x: len(x.split()))
    merged["pred_length_bucket"] = merged["pred_word_count"].apply(assign_length_bucket)

    pred_length_analysis = merged.groupby("pred_length_bucket").agg(
        count=("rougeL", "count"),
        avg_rougeL=("rougeL", "mean"),
        avg_bertscore=("bertscore_f1", "mean"),
    ).round(4)
    print(pred_length_analysis.to_string(), flush=True)

    # Save results
    results = {
        "ROUGE-L": rougeL,
        "BERTScore": bertscore_f1,
        "num_examples": len(merged),
        "by_length": length_analysis.to_dict(),
        "by_prompt_type": type_analysis.to_dict(),
    }

    output_file = args.output_file or args.prediction_file.replace(".tsv", "_eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}", flush=True)


if __name__ == "__main__":
    main()