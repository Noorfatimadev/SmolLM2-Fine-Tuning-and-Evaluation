#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/fp/projects01/ec403/IN5550/obligatories/3}"
OUT_DIR="${OUT_DIR:-results_eda}"

python3 eda_oblig3.py \
  --train "$DATA_DIR/obli3_train.tsv.gz" \
  --test "$DATA_DIR/obli3_test.tsv.gz" \
  --out_dir "$OUT_DIR"
