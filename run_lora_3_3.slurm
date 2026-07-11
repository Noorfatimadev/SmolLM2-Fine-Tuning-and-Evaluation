#!/bin/bash
#SBATCH --job-name=oblig3_lora
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results_lora/slurm_%j.out
#SBATCH --error=results_lora/slurm_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results_lora

module purge
module load nlpl-trl/0.22.2-foss-2024a-Python-3.12.3

export PYTHONNOUSERSITE=1
export NO_TORCH_COMPILE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/fp/projects01/ec403/hf_models
export TRANSFORMERS_CACHE=/fp/projects01/ec403/hf_models/hub
export HF_HUB_OFFLINE=1

DATA_DIR=/fp/projects01/ec403/IN5550/obligatories/3
MODEL=HuggingFaceTB/SmolLM2-1.7B
RUN_DIR=results_lora/smollm2_1_7b_lora_r16

python3 train_lora_oblig3.py \
  --train_path "$DATA_DIR/obli3_train.tsv.gz" \
  --max_train_examples 5000 \
  --model_name_or_path "$MODEL" \
  --output_dir "$RUN_DIR" \
  --epochs 1 --max_train_examples 3000 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_length 1024

python3 predict_lora_oblig3.py \
  --input_path "$DATA_DIR/obli3_test.tsv.gz" \
  --output_path "$RUN_DIR/test_predictions.tsv" \
  --base_model "$MODEL" \
  --adapter_path "$RUN_DIR/adapter" \
  --batch_size 4 \
  --max_new_tokens 128

python3 eval_generation.py \
  --pred_path "$RUN_DIR/test_predictions.tsv" \
  --gold_path "$DATA_DIR/obli3_test.tsv.gz" \
  --out_path "$RUN_DIR/test_metrics.csv"
