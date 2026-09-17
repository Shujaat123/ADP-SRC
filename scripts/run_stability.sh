#!/usr/bin/env bash
set -euo pipefail
python -m adp_src.stability_evaluation \
  --data data/positive_negative.fasta \
  --output-dir results/stability_direct_5x5 \
  --protocol direct \
  --folds 5 \
  --seeds 6 17 29 41 53
python -m adp_src.stability_evaluation \
  --data data/positive_negative.fasta \
  --output-dir results/stability_cluster_5x5 \
  --protocol cluster \
  --cluster-file data/positive_negative_cdhit50.fasta.clstr \
  --folds 5 \
  --seeds 6 17 29 41 53
