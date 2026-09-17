#!/usr/bin/env bash
set -euo pipefail
python -m adp_src.lse_evaluation \
  --data data/positive_negative.fasta \
  --output-dir results/direct_5fold \
  --protocol direct \
  --folds 5 \
  --seed 6 \
  --epochs 100 \
  --patience 14 \
  --lse-features pearson347 \
  --device auto
