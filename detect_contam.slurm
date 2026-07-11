#!/bin/bash
#SBATCH --job-name=detect_contam
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --gpus=a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/detect_contam_%j.out
#SBATCH --error=logs/detect_contam_%j.err

mkdir -p logs

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-trl/0.22.2-foss-2024a-Python-3.12.3

cd /cluster/work/projects/ec403/ec-mykhailk/Assigment3

python detect_contamination.py \
    --train_file obli3_train.tsv.gz \
    --test_file obli3_test.tsv.gz \
    --output_file obli3_train_clean.tsv.gz \
    --ngram 13 \
    --threshold 0.8 \
    --report_file contamination_report.json
