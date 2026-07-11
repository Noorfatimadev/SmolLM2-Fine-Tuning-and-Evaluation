#!/bin/bash
#SBATCH --job-name=generate_preds
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --gpus=a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err

mkdir -p logs

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-trl/0.22.2-foss-2024a-Python-3.12.3

cd /cluster/work/projects/ec403/ec-mykhailk/Assigment3

# Generate predictions from the base model
echo "=== Generating predictions: SmolLM2-135M (base) ==="
python generate_predictions.py \
    --model_dir ./output_base_135M \
    --test_file obli3_test.tsv.gz \
    --output_file predictions_base_135M.tsv \
    --max_new_tokens 512 \
    --batch_size 16

# Generate predictions from the instruct model
echo "=== Generating predictions: SmolLM2-135M-Instruct ==="
python generate_predictions.py \
    --model_dir ./output_instruct_135M \
    --test_file obli3_test.tsv.gz \
    --output_file predictions_instruct_135M.tsv \
    --max_new_tokens 512 \
    --batch_size 16

echo "Done!"
