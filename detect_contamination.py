"""
detect_contamination.py — Detect contamination between train and test sets
using exact match and n-gram overlap (Jaccard similarity on n-grams).

Usage:
    python detect_contamination.py \
        --train_file obli3_train.tsv.gz \
        --test_file obli3_test.tsv.gz \
        --output_file obli3_train_clean.tsv.gz \
        --ngram 13 \
        --threshold 0.8
"""

import os
import gzip
import pandas as pd
from argparse import ArgumentParser
from collections import defaultdict


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to write the cleaned training set")
    parser.add_argument("--ngram", type=int, default=13,
                        help="N-gram size (13 is a common choice from GPT-3 paper)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Jaccard similarity threshold for n-gram contamination")
    parser.add_argument("--report_file", type=str, default="contamination_report.json")
    return parser.parse_args()


def normalize(text):
    "Lowercase and collapse whitespace for comparison."
    return " ".join(str(text).lower().split())


def get_ngrams(text, n):
    "Return the set of word-level n-grams in the text."
    tokens = normalize(text).split()
    if len(tokens) < n:
        return set()
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def main():
    args = parse_args()

    print("Loading data...", flush=True)
    train = pd.read_csv(args.train_file, sep="\t")
    test = pd.read_csv(args.test_file, sep="\t")
    train = train.dropna(subset=["prompt", "response"]).reset_index(drop=True)
    test = test.dropna(subset=["prompt", "response"]).reset_index(drop=True)
    print(f"Train examples: {len(train)}", flush=True)
    print(f"Test examples: {len(test)}", flush=True)

    #  1) Exact-match contamination
    print("\n=== Exact-match detection ===", flush=True)
    test_prompts_norm = set(test["prompt"].apply(normalize))
    train["prompt_norm"] = train["prompt"].apply(normalize)
    train["exact_match"] = train["prompt_norm"].isin(test_prompts_norm)
    n_exact = train["exact_match"].sum()
    print(f"Exact-match contaminated training examples: {n_exact} "
          f"({100 * n_exact / len(train):.2f}%)", flush=True)

    # 2) N-gram / Jaccard contamination
    print(f"\n=== N-gram detection (n={args.ngram}, threshold={args.threshold}) ===",
          flush=True)

    print("Building test n-gram index...", flush=True)
    test_ngrams = [get_ngrams(p, args.ngram) for p in test["prompt"]]

    # Inverted index: n-gram -> set of test indices that contain it
    ngram_to_test = defaultdict(set)
    for i, ngrams in enumerate(test_ngrams):
        for ng in ngrams:
            ngram_to_test[ng].add(i)

    print("Scanning training set for n-gram contamination...", flush=True)
    ngram_contaminated = []
    for i, prompt in enumerate(train["prompt"]):
        if i % 5000 == 0:
            print(f"  {i}/{len(train)}", flush=True)
        train_ngrams = get_ngrams(prompt, args.ngram)
        if not train_ngrams:
            ngram_contaminated.append(False)
            continue
        # Candidate test examples are those sharing at least one n-gram
        candidates = set()
        for ng in train_ngrams:
            candidates.update(ngram_to_test.get(ng, set()))
        # Compute Jaccard with each candidate and check threshold
        is_contaminated = False
        for cand in candidates:
            if jaccard(train_ngrams, test_ngrams[cand]) >= args.threshold:
                is_contaminated = True
                break
        ngram_contaminated.append(is_contaminated)

    train["ngram_match"] = ngram_contaminated
    n_ngram = train["ngram_match"].sum()
    print(f"N-gram contaminated training examples: {n_ngram} "
          f"({100 * n_ngram / len(train):.2f}%)", flush=True)

    # Combined contamination
    train["contaminated"] = train["exact_match"] | train["ngram_match"]
    n_total = train["contaminated"].sum()
    print(f"\nTotal unique contaminated examples: {n_total} "
          f"({100 * n_total / len(train):.2f}%)", flush=True)

    # Overlap between the two methods
    n_both = (train["exact_match"] & train["ngram_match"]).sum()
    print(f"Examples flagged by BOTH methods: {n_both}", flush=True)
    print(f"Exact-only: {n_exact - n_both}", flush=True)
    print(f"N-gram-only: {n_ngram - n_both}", flush=True)

    # Write cleaned training set
    clean = train[~train["contaminated"]][["prompt", "response"]]
    print(f"\nCleaned training set size: {len(clean)} "
          f"(removed {len(train) - len(clean)})", flush=True)

    clean.to_csv(args.output_file, sep="\t", index=False, compression="gzip")
    print(f"Cleaned training set saved to {args.output_file}", flush=True)

    # Save contamination report
    import json
    report = {
        "train_size_original": len(train),
        "test_size": len(test),
        "exact_match_contaminated": int(n_exact),
        "ngram_match_contaminated": int(n_ngram),
        "ngram_size": args.ngram,
        "ngram_threshold": args.threshold,
        "total_contaminated_unique": int(n_total),
        "flagged_by_both": int(n_both),
        "exact_only": int(n_exact - n_both),
        "ngram_only": int(n_ngram - n_both),
        "train_size_cleaned": len(clean),
    }
    with open(args.report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Contamination report saved to {args.report_file}", flush=True)


if __name__ == "__main__":
    main()
