# Assignment3_NLP


## Data

Assignment data files used in this repository:

- `obli3_train.tsv.gz` - Task 3.2 
- `obli3_train_clean.tsv.gz` Task 3.5
- `obli3_test.tsv.gz` Task 3.2 


These files are the main datasets for training, evaluation, contamination checking, and final held-out prediction generation.

---

## Training

### Main training script

- `train_sft.py`

This is the main supervised fine-tuning script used to train the models for Tasks 3.2 and 3.5

### SLURM training scripts

- `train_base.slurm` - Task 3.2 
- `train_base_clean.slurm` - Task 3.5
- `train_instruct.slurm` - Task 3.2 

These scripts launch the different training configurations on the cluster:
- `train_base.slurm`: trains the base setup - Task 3.2 
- `train_base_clean.slurm`: trains on the cleaned training dataset - Task 3.5
- `train_instruct.slurm`: trains the instruction-tuned setup - Task 3.2  

---

## Generation

### Main generation script

- `generate_predictions.py`

This script loads a trained model checkpoint and generates predictions in `.tsv` format.

### SLURM generation scripts

- `generate.slurm` - Task 3.2 
- `generate_instruct.slurm` - Task 3.2 
- `gen_eval_clean.slurm`- Task 3.5

These scripts are used for prediction generation in different setups:
- `generate.slurm`: standard prediction generation - Task 3.2 
- `generate_instruct.slurm`: prediction generation for the instruction-tuned model - Task 3.2 
- `gen_eval_clean.slurm`: generation/evaluation pipeline for the clean-data setup - Task 3.5
---

## Evaluation

### Main evaluation scripts

- `eval_on_test.py` - Provided eval script
- `evaluate_metrics.py` - Task 3.2, 3.5 

These scripts are used to evaluate generated predictions:
- `eval_on_test.py`: evaluates prediction files in the required tab-separated format
- `evaluate_metrics.py`: computes metrics such as ROUGE-L and BERTScore

### SLURM evaluation scripts

- `evaluate.slurm` - Task 3.2
- `eval_clean.slurm` - Task 3.5

These scripts run evaluation jobs on the cluster:
- `evaluate.slurm`: standard evaluation
- `eval_clean.slurm`: evaluation for the clean-data setup

---

## Contamination checking

### Files

- `detect_contamination.py` - Task 3.5
- `detect_contam.slurm` - Task 3.5
- `contamination_report.json` - Task 3.5

These files are used to check for possible contamination between datasets:
- `detect_contamination.py`: script for detecting overlap or leakage
- `detect_contam.slurm`: SLURM job file for running contamination detection
- `contamination_report.json`: output report from the contamination check

---

## Best model

### Best configuration used

- Model adapter: `results_lora/smollm2_1_7b_lora_r16/adapter`
- Base model: `HuggingFaceTB/SmolLM2-1.7B`

### Test metrics

- **ROUGE-L**: `0.2164`
- **BERTScore F1**: `0.8606`

This model was selected as the best-performing model and used to generate the held-out predictions.

---

## Prediction files

Heldout predictions for the best model located here: `results_lora/smollm2_1_7b_lora_r16/heldout_predictions.tsv`

---

## Evaluation result files

- `predictions_base_135M_eval_results.json`
- `predictions_base_135M_clean_eval_results.json`
- `predictions_instruct_135M_eval_results.json`

These files store the evaluation results for the corresponding prediction files.

---

## Minimal reproducible commands for Task 3.2

```bash
# Train
sbatch train_base.slurm
sbatch train_instruct.slurm

# Generate predictions
sbatch generate.slurm
sbatch generate_instruct.slurm


# Evaluate predictions
sbatch evaluate.slurm
```

## Minimal reproducible commands for Task 3.5

```bash
# Run contamination detection
sbatch detect_contam.slurm

# Review contamination report
cat contamination_report.json

# Train on cleaned data
sbatch train_base_clean.slurm

# Generate predictions from the cleaned-data model
sbatch gen_eval_clean.slurm
```

