#!/bin/bash
#SBATCH --job-name=sft_base_clean
#SBATCH --account=ec403
#SBATCH --partition=accel
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/train_base_clean_%j.out
#SBATCH --error=logs/train_base_clean_%j.err

mkdir -p logs

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-trl/0.22.2-foss-2024a-Python-3.12.3

cd /cluster/work/projects/ec403/ec-mykhailk/Assigment3

# Same hyperparameters as in Section 3.2, but training on the cleaned data
python train_sft.py \
    --model_name HuggingFaceTB/SmolLM2-135M \
    --train_file obli3_train_clean.tsv.gz \
    --output_dir ./output_base_135M_clean \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_seq_length 1024 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --bf16
