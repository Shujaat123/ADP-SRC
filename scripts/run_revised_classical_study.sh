#!/usr/bin/env bash
set -euo pipefail

python -m adp_src.direct_comparator_stability \
  --data data/positive_negative.fasta \
  --output-dir results/revised_direct_comparators_5x5 \
  --folds 5 --seeds 6 17 29 41 53

python -m adp_src.pca_ablation \
  --data data/positive_negative.fasta \
  --output-dir results/revised_pca_dimension_ablation_extended_5x5 \
  --dimensions 16 32 64 96 128 160 192 224 240 \
  --folds 5 --seeds 6 17 29 41 53

python -m adp_src.pca_ablation \
  --data data/positive_negative.fasta \
  --output-dir results/revised_pca160_cdhit50_5x5 \
  --dimensions 160 \
  --protocol cluster \
  --cluster-file data/positive_negative_cdhit50.fasta.clstr \
  --folds 5 --seeds 6 17 29 41 53
