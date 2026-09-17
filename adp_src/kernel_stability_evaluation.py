"""Repeated-CV partition stability for the locked kernel SRC specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .data import read_labeled_fasta, records_to_arrays
from .features import extract_cksaap
from .kernel_src import PolynomialCKSAAPEmbedding, l1_src_predict_proba
from .metrics import classification_metrics
from .splits import make_outer_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/positive_negative.fasta")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=("direct", "cluster"), required=True)
    parser.add_argument("--cluster-file")
    parser.add_argument("--seeds", nargs="+", type=int, default=[6, 17, 29, 41, 53])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.001)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_labeled_fasta(args.data)
    sequences, y, identifiers = records_to_arrays(records)
    raw = extract_cksaap(sequences, max_gap=8)
    repeat_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for repeat, seed in enumerate(args.seeds):
        splits, _ = make_outer_splits(
            y, identifiers, args.protocol, seed, args.folds, args.cluster_file
        )
        probability = np.full(len(y), np.nan)
        for fold, (train, test) in enumerate(splits):
            embedding = PolynomialCKSAAPEmbedding(
                n_components=args.dimension, degree=2, seed=seed + 100 * fold
            ).fit(raw[train], y[train])
            train_z = normalize(embedding.transform(raw[train]), axis=1)
            test_z = normalize(embedding.transform(raw[test]), axis=1)
            fold_probability, _, _ = l1_src_predict_proba(
                train_z, y[train], test_z, alpha=args.alpha
            )
            probability[test] = fold_probability
            fold_rows.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "fold": fold,
                    "method": "CKSAAP-KPCA-L1-SRC",
                    **classification_metrics(y[test], fold_probability),
                }
            )
        repeat_rows.append(
            {
                "repeat": repeat,
                "seed": seed,
                "method": "CKSAAP-KPCA-L1-SRC",
                **classification_metrics(y, probability),
            }
        )
        print(f"completed repeat {repeat + 1}/{len(args.seeds)} (seed={seed})", flush=True)
    repeat_frame = pd.DataFrame(repeat_rows)
    repeat_frame.to_csv(output / "repeat_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    metrics = ["accuracy", "sensitivity", "specificity", "mcc", "f1", "auc_roc"]
    aggregate = repeat_frame.groupby("method")[metrics].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output / "aggregate_metrics.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": args.protocol,
                "folds": args.folds,
                "seeds": args.seeds,
                "dimension": args.dimension,
                "alpha": args.alpha,
                "hyperparameters_locked_before_repeated_analysis": True,
                "interpretation": "Partition-stability analysis; repeats share observations and are not independent external tests.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(aggregate.to_string())


if __name__ == "__main__":
    main()
