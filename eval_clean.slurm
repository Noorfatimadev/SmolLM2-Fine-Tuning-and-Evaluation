#!/bin/bash
#SBATCH --job-name=eval_clean
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --gpus=a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/eval_clean_%j.out
#SBATCH --error=logs/eval_clean_%j.err

mkdir -p logs
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-trl/0.22.2-foss-2024a-Python-3.12.3

cd /cluster/work/projects/ec403/ec-mykhailk/Assigment3

python evaluate_metrics.py \
    --prediction_file predictions_base_135M_clean.tsv \
    --gold_file obli3_test.tsv.gz \
    --batch_size 32
