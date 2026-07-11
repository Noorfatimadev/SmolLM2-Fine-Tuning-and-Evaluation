#!/usr/bin/env python3
"""EDA for IN5550 Obligatory 3, section 3.1.

This script reads the train/test TSV files, computes response length
distributions and instruction-type distributions, and writes CSV tables and
plots for the report.
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("summarize", re.compile(r"\b(summarize|summarise|summary|sum up)\b", re.I)),
    ("translate", re.compile(r"\b(translate|translation)\b", re.I)),
    ("write", re.compile(r"\b(write|compose|draft|create|generate)\b", re.I)),
    ("classify", re.compile(r"\b(classify|categorize|label|identify whether)\b", re.I)),
    ("analyze", re.compile(r"\b(analyze|analyse|compare|explain|discuss|evaluate)\b", re.I)),
    ("reasoning", re.compile(r"\b(calculate|solve|reason|if|why|how many|what is)\b", re.I)),
]


def length_bin_edges(max_tokens: int) -> list[int]:
    upper = max(100, ((max_tokens // 100) + 2) * 100)
    return list(range(0, upper + 1, 100))


def read_tsv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return pd.read_csv(f, sep="\t")
    return pd.read_csv(path, sep="\t")


def token_count(text: object) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\S+", text))


def instruction_type(prompt: object) -> str:
    if not isinstance(prompt, str):
        return "other"
    for label, pattern in TYPE_PATTERNS:
        if pattern.search(prompt):
            return label
    return "other"


def add_features(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    df["split"] = split
    df["response_tokens"] = df["response"].apply(token_count)
    df["prompt_tokens"] = df["prompt"].apply(token_count)
    df["instruction_type"] = df["prompt"].apply(instruction_type)
    df["response_length_bin"] = pd.cut(
        df["response_tokens"],
        bins=length_bin_edges(int(df["response_tokens"].max())),
        right=False,
        include_lowest=True,
    ).astype(str)
    return df


def length_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in df.groupby("split", sort=False):
        for bin_name, bin_group in group.groupby("response_length_bin", observed=False):
            rows.append(
                {
                    "split": split,
                    "response_length_bin": bin_name,
                    "count": len(bin_group),
                    "percentage": round(100 * len(bin_group) / len(group), 2),
                }
            )
    return pd.DataFrame(rows)


def length_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in df.groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "n_examples": len(group),
                "mean_response_tokens": round(group["response_tokens"].mean(), 2),
                "median_response_tokens": round(group["response_tokens"].median(), 2),
                "p90_response_tokens": round(group["response_tokens"].quantile(0.90), 2),
                "max_response_tokens": int(group["response_tokens"].max()),
            }
        )
    return pd.DataFrame(rows)


def type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in df.groupby("split", sort=False):
        counts = group["instruction_type"].value_counts().sort_index()
        for label, count in counts.items():
            rows.append(
                {
                    "split": split,
                    "instruction_type": label,
                    "count": int(count),
                    "percentage": round(100 * count / len(group), 2),
                }
            )
    return pd.DataFrame(rows)


def plot_response_lengths(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for split in ["train", "test", "full"]:
        subset = df[df["split"] == split]
        plt.hist(
            subset["response_tokens"],
            bins=length_bin_edges(int(df["response_tokens"].max())),
            alpha=0.45,
            label=split,
        )
    plt.xlabel("Response length (whitespace tokens)")
    plt.ylabel("Number of examples")
    plt.title("Response length distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_instruction_types(type_df: pd.DataFrame, out_path: Path) -> None:
    pivot = type_df.pivot(index="instruction_type", columns="split", values="percentage").fillna(0)
    ordered_cols = [col for col in ["train", "test", "full"] if col in pivot.columns]
    pivot[ordered_cols].plot(kind="bar", figsize=(10, 6))
    plt.xlabel("Instruction type")
    plt.ylabel("Percentage of split")
    plt.title("Instruction type distribution")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("results_eda"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = add_features(read_tsv(args.train), "train")
    test = add_features(read_tsv(args.test), "test")
    full = add_features(pd.concat([train.drop(columns=["split"]), test.drop(columns=["split"])]), "full")
    all_df = pd.concat([train, test, full], ignore_index=True)

    length_dist = length_distribution(all_df)
    length_stats = length_summary(all_df)
    type_dist = type_distribution(all_df)

    length_dist.to_csv(args.out_dir / "response_length_distribution.csv", index=False)
    length_stats.to_csv(args.out_dir / "response_length_summary.csv", index=False)
    type_dist.to_csv(args.out_dir / "instruction_type_distribution.csv", index=False)
    all_df[["split", "prompt_tokens", "response_tokens", "instruction_type"]].to_csv(
        args.out_dir / "eda_examples_with_features.csv", index=False
    )

    plot_response_lengths(all_df, args.out_dir / "response_length_distribution.png")
    plot_instruction_types(type_dist, args.out_dir / "instruction_type_distribution.png")

    print("Saved EDA outputs to:", args.out_dir)
    print("\nResponse length summary:")
    print(length_stats.to_string(index=False))
    print("\nInstruction type distribution:")
    print(type_dist.to_string(index=False))


if __name__ == "__main__":
    main()
