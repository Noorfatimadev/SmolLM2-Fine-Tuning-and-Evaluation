#!/usr/bin/env python3
"""Evaluate generated text with ROUGE-L and BERTScore."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def tokens(text: str) -> list[str]:
    return re.findall(r"\S+", str(text).lower())


def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0]
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_toks = tokens(prediction)
    ref_toks = tokens(reference)
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = lcs_len(pred_toks, ref_toks)
    precision = lcs / len(pred_toks)
    recall = lcs / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=Path, required=True)
    parser.add_argument("--gold_path", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, default=Path("results_lora/eval_metrics.csv"))
    parser.add_argument("--bertscore_model", default="/fp/projects01/ec403/hf_models/roberta-large")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(args.pred_path, sep="\t")
    gold_df = pd.read_csv(args.gold_path, sep="\t")
    preds = pred_df["prediction"].astype(str).tolist()
    refs = gold_df["response"].astype(str).tolist()

    if len(preds) != len(refs):
        raise ValueError(f"Length mismatch: {len(preds)} predictions vs {len(refs)} references")

    rouge_scores = [rouge_l_f1(pred, ref) for pred, ref in zip(preds, refs)]
    result = {
        "n_examples": len(preds),
        "rouge_l": sum(rouge_scores) / len(rouge_scores),
    }

    try:
        from bert_score import score

        _, _, f1 = score(
            preds,
            refs,
            model_type=str(args.bertscore_model),
            batch_size=args.batch_size,
            verbose=True,
            lang="en",
            rescale_with_baseline=False,
        )
        result["bertscore_f1"] = float(f1.mean().item())
    except Exception as exc:
        result["bertscore_f1"] = float("nan")
        result["bertscore_error"] = str(exc)

    out = pd.DataFrame([result])
    out.to_csv(args.out_path, index=False)
    print(out.to_string(index=False))
    print("Saved metrics to:", args.out_path)


if __name__ == "__main__":
    main()
