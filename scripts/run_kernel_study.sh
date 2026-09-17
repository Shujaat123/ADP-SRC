#!/usr/bin/env bash
set -euo pipefail
python -m adp_src.kernel_evaluation \
  --data data/positive_negative.fasta \
  --output-dir results/kernel_direct_5fold \
  --protocol direct --seed 6 --folds 5
python -m adp_src.kernel_evaluation \
  --data data/positive_negative.fasta \
  --output-dir results/kernel_cluster_5fold \
  --protocol cluster --seed 6 --folds 5 \
  --cluster-file data/positive_negative_cdhit50.fasta.clstr
