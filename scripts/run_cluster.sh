#!/usr/bin/env bash
set -euo pipefail
python -m adp_src.lse_evaluation \
  --data data/positive_negative.fasta \
  --output-dir results/cluster_5fold \
  --protocol cluster \
  --cluster-file data/positive_negative_cdhit50.fasta.clstr \
  --folds 5 \
  --seed 6 \
  --epochs 100 \
  --patience 14 \
  --lse-features pearson347 \
  --device auto
