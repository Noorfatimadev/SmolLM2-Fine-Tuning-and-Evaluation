#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --gpus=a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

mkdir -p logs

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-trl/0.22.2-foss-2024a-Python-3.12.3

cd /cluster/work/projects/ec403/ec-mykhailk/Assigment3

  # Evaluate the base model
echo "=== Evaluating SmolLM2-135M (base) ==="
python evaluate_metrics.py \
    --prediction_file predictions_base_135M.tsv \
    --gold_file obli3_test.tsv.gz \
    --batch_size 32

# Evaluate the instruct model
echo "=== Evaluating SmolLM2-135M-Instruct ==="
python evaluate_metrics.py \
    --prediction_file predictions_instruct_135M.tsv \
    --gold_file obli3_test.tsv.gz \
    --batch_size 32

echo "Done!"
