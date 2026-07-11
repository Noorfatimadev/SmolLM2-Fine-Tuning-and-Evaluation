import os

CACHE_DIR = "/cluster/work/projects/ec403/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_METRICS_CACHE"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

import re
import json
import torch
import evaluate
from argparse import ArgumentParser
import transformers
import numpy as np
import pandas as pd
from datasets import Dataset


def parse_args():
    """
    Parses the arguments required for the evaluation script.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--prediction_fpath",
        required=True,
        type=str,
        help="Path to a tab-separated dataframe containing the best system's predictions.",
    )
    parser.add_argument(
        "--gold_fpath",
        required=False,
        type=str,
        help="Path to the private test set with the human references.",
    )
    parser.add_argument(
        "--judge_prompts_fpath",
        required=False,
        type=str,
        default="judge_prompts.json",
        help="Path to the judge prompts.",
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=8,
        help="A batch size for the BERTScore and judge model.",
    )
    parser.add_argument(
        "--judge_model_name",
        required=False,
        type=str,
        default="/fp/projects01/ec403/hf_models/Qwen3-8B",
        help="The name of the judge model.",
    )
    args = parser.parse_args()
    return args


def load_judge_prompts(fpath):
    """
    Loads the judge prompt from a JSON file.
    args:
        fpath (str): The path to the JSON file containing the judge prompt.
    returns:
        dict: A dictionary containing the judge prompt.
    """
    with open(fpath, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts


def save_results(results, fname):
    """
    Saves the results into a .json file.
    args:
        results (dictionary): A dictionary containing the performance results.
        fname (string): A name of the output file.
    """
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(results, f)


def create_pairwise_comparison(example, prompt_template):
    """
    Creates an input example for the baseline based on the prompt.
    args:
        example (dataframe series): A series containing the instruction, a system prediction, and a human reference.
        prompt_template (str): The prompt template.
    returns:
        string: Formatted prompt template for the judge.
    """
    example["pairwise_comparison"] = prompt_template.format(
        prompt=example["prompt"],
        answer_a=example["prediction"],
        answer_b=example["response"],
    )
    return example


def postprocess_pairwise_comparison(
    pairwise_comparison,
    fallback="[[C]]",
    mapping={"[[A]]": 1, "[[B]]": 0, "[[C]]": 0.5},
):
    """
    Extracts the judge's verdict. If the verdict cannot be automatically extracted, we use the "tie" class as the fallback.
    args:
        pairwise_comparison (str): An output from the judge.
        fallback (string): A default answer if the judge's verdict cannot be extracted.
    returns:
        string: the judge's verdict for a given example.
    """
    prediction = pairwise_comparison[0]["generated_text"]  # [-1]["content"]
    verdict = re.search(r"\[[^\w\s]*\[\s*[ABC]\s*\][^\w\s]*\]", prediction)
    verdict = verdict.group(0) if verdict is not None else fallback
    return mapping[verdict]


def compute_rouge(predictions, references, rouge_metric):
    """
    Computes the ROUGE-L score between a list of system-generated predictions and human references.
    args:
        predictions (list of str): A list of system-generated outputs.
        references (list of str): A list of human references.
        rouge_metric (evaluate.metrics.rouge.Rouge): An instance of the `evaluate` ROUGE metric.
    returns:
        float: The ROUGE-L score as a percentage, rounded to three decimal places.
    """
    rougeL = rouge_metric.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rougeL"],
    )["rougeL"]
    rougeL_avg = round(rougeL * 100, 3)
    return rougeL_avg


def compute_bertscore(predictions, references, bertscore_metric, batch_size):
    """
    Computes the BERTScore between a list of system and human explanations.
    args:
        predictions (list of str): A list of system-generated outputs.
        references (list of str): A list of human references.
        bertscore_metric (evaluate.metrics.bertscore.BERTScore): An instance of the `evaluate` BERTScore metric.
        batch_size (int): The batch size.
    returns:
        float: The BERTScore as a percentage, rounded to three decimal places.
    """
    print("Computing the BERTScore...", flush=True)
    bertscore_f1 = bertscore_metric.compute(
        predictions=predictions, references=references, lang="en", batch_size=batch_size
    )["f1"]
    bertscore_f1_avg = round(np.mean(bertscore_f1) * 100, 3)
    return bertscore_f1_avg


def compute_judge_score(dataset, pipeline, system_prompt, batch_size):
    """
    Computes the win-rate for the system against the human without accounting for a position bias.
    args:
        dataset (datasets.arrow_dataset.Dataset): The dataset containing formatted inputs for the judge model.
        pipeline (transformers.pipeline): The judge model.
        system_prompt (str): The system prompt.
        batch_size (int): The batch size.
    returns:
        float: The win-rate value.
    """
    tokenizer = pipeline.tokenizer
    messages = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pairwise_comparison},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for pairwise_comparison in dataset["pairwise_comparison"]
    ]
    print("Computing the win-rate...\n\nExample: {}".format(messages[0]), flush=True)
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.9,
        batch_size=batch_size,
        return_full_text=False,
    )
    win_rate = np.mean(list(map(postprocess_pairwise_comparison, outputs)))
    win_rate = round(win_rate * 100, 3)
    return win_rate


def run_evaluation(dataset, judge_model_name, batch_size):
    """
    Computes the language generation evaluation metrics and the win-rate based on the LLM-as-a-judge.
    args:
        dataset (datasets.arrow_dataset.Dataset): The dataset containing formatted inputs for the judge model.
        judge_model_name (str): The name of the judge model.
    returns:
        dictionary: A dictionary with the performance scores.
    """
    results = {}
    # compute the ROUGE-L score
    rouge_metric = evaluate.load("rouge")
    rouge_score = compute_rouge(
        predictions=dataset["prediction"],
        references=dataset["response"],
        rouge_metric=rouge_metric,
    )
    results["ROUGE-L"] = rouge_score
    # compute the BERTScore
    bertscore_metric = evaluate.load(
        "bertscore",
        cache_dir=CACHE_DIR,
    )
    bertscore = compute_bertscore(
        predictions=dataset["prediction"],
        references=dataset["response"],
        bertscore_metric=bertscore_metric,
        batch_size=batch_size,
    )
    results["BERTScore"] = bertscore
    # saving RAM
    del bertscore_metric
    # compute the win-rate against the humans
    pipeline = transformers.pipeline(
        "text-generation",
        model=judge_model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "cache_dir": CACHE_DIR,
        },
        device_map="auto",
    )
    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    win_rate = compute_judge_score(
        dataset=dataset,
        pipeline=pipeline,
        system_prompt=system_prompt,
        batch_size=batch_size,
    )
    results["LLM-as-a-judge"] = win_rate
    return results


if __name__ == "__main__":
    args = parse_args()
    # loading the judge prompts
    prompts = load_judge_prompts(args.judge_prompts_fpath)
    prompt_template = prompts["prompt_template"]
    system_prompt = prompts["system_prompt"]
    # preparing the dataset
    test = pd.read_csv(args.gold_fpath, sep="\t")
    predictions = pd.read_csv(args.prediction_fpath, sep="\t")

    # Explicit merge on prompt
    merged = test.merge(predictions, on="prompt", how="inner")

    # Remove rows with missing values
    merged = merged.dropna(subset=["response", "prediction"])

    # Convert to string and strip spaces
    merged["response"] = merged["response"].astype(str).str.strip()
    merged["prediction"] = merged["prediction"].astype(str).str.strip()

    # Remove empty rows
    merged = merged[(merged["response"] != "") & (merged["prediction"] != "")]

    print("Merged examples after cleaning:", len(merged), flush=True)

    dataset = Dataset.from_pandas(merged).map(
        create_pairwise_comparison,
        fn_kwargs={"prompt_template": prompt_template},
        batched=False,
    )
    results = run_evaluation(
        dataset=dataset,
        judge_model_name=args.judge_model_name,
        batch_size=args.batch_size,
    )
    print("Results: {}".format(results), flush=True)
    out_fpath = args.prediction_fpath.replace(".tsv", "_results.json")
    print("Saving results to {}".format(out_fpath), flush=True)
    save_results(results=results, fname=out_fpath)
