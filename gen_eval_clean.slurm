#!/bin/bash
#SBATCH --job-name=gen_eval_clean
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gen_eval_clean_%j.out
#SBATCH --error=logs/gen_eval_clean_%j.err

mkdir -p logs

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-trl/0.22.2-foss-2024a-Python-3.12.3

cd /cluster/work/projects/ec403/ec-mykhailk/Assigment3

# Generate predictions from the cleaned-data model
echo "=== Generating predictions: SmolLM2-135M (cleaned data) ==="
python generate_predictions.py \
    --model_dir ./output_base_135M_clean \
    --test_file obli3_test.tsv.gz \
    --output_file predictions_base_135M_clean.tsv \
    --max_new_tokens 512 \
    --batch_size 8

# Evaluate
echo "=== Evaluating SmolLM2-135M (cleaned data) ==="
python evaluate_metrics.py \
    --prediction_file predictions_base_135M_clean.tsv \
    --gold_file obli3_test.tsv.gz \
    --batch_size 32

echo "Done!"
