#!/bin/bash
#SBATCH --job-name=oblig3_speed
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --gres=gpu:rtx30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=results_inference_speed/slurm_%j.out
#SBATCH --error=results_inference_speed/slurm_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results_inference_speed

module purge
module load nlpl-vllm/0.8.2-foss-2024a-Python-3.12.3

export PYTHONNOUSERSITE=1
export NO_TORCH_COMPILE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/fp/projects01/ec403/hf_models
export TRANSFORMERS_CACHE=/fp/projects01/ec403/hf_models/hub
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

DATA_DIR=/fp/projects01/ec403/IN5550/obligatories/3
MODEL=HuggingFaceTB/SmolLM2-1.7B
N_EXAMPLES="${N_EXAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

python3 benchmark_inference_3_4.py \
  --input_path "$DATA_DIR/obli3_test.tsv.gz" \
  --out_dir results_inference_speed \
  --backend hf \
  --model_name_or_path "$MODEL" \
  --max_examples "$N_EXAMPLES" \
  --batch_size 4 \
  --max_new_tokens "$MAX_NEW_TOKENS"

python3 benchmark_inference_3_4.py \
  --input_path "$DATA_DIR/obli3_test.tsv.gz" \
  --out_dir results_inference_speed \
  --backend vllm \
  --model_name_or_path "$MODEL" \
  --max_examples "$N_EXAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS"

python3 - <<'PY'
from pathlib import Path
import pandas as pd

out = Path("results_inference_speed")
rows = []
for name in ["hf_speed.csv", "vllm_speed.csv"]:
    path = out / name
    if path.exists():
        rows.append(pd.read_csv(path))

summary = pd.concat(rows, ignore_index=True)
summary.to_csv(out / "speed_comparison.csv", index=False)
print(summary.to_string(index=False))
print("Saved:", out / "speed_comparison.csv")
PY
