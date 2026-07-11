#!/bin/bash
#SBATCH --job-name=gen_instruct
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gen_instruct_%j.out
#SBATCH --error=logs/gen_instruct_%j.err

mkdir -p logs
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-trl/0.22.2-foss-2024a-Python-3.12.3

cd /cluster/work/projects/ec403/ec-mykhailk/Assigment3

python generate_predictions.py \
    --model_dir ./output_instruct_135M \
    --test_file obli3_test.tsv.gz \
    --output_file predictions_instruct_135M.tsv \
    --max_new_tokens 512 \
    --batch_size 8
