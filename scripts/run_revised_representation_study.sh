#!/usr/bin/env bash
set -euo pipefail

python -m adp_src.representation_stability \
  --data data/positive_negative.fasta \
  --output-dir results/revised_src_variants_160_5x5 \
  --dimension 160 \
  --folds 5 --seeds 6 17 29 41 53 \
  --epochs 100 --patience 14 --device auto
