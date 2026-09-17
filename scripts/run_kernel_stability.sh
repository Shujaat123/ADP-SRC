#!/usr/bin/env bash
set -euo pipefail
python -m adp_src.kernel_stability_evaluation \
  --data data/positive_negative.fasta \
  --output-dir results/kernel_stability_direct_5x5 \
  --protocol direct --seeds 6 17 29 41 53
python -m adp_src.kernel_stability_evaluation \
  --data data/positive_negative.fasta \
  --output-dir results/kernel_stability_cluster_5x5 \
  --protocol cluster \
  --cluster-file data/positive_negative_cdhit50.fasta.clstr \
  --seeds 6 17 29 41 53
