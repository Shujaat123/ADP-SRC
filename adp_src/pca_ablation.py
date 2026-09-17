"""Repeated direct-CV ablation of the PCA dimension used by SRC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .classical import SRCTransformer, src_predict_proba
from .data import read_labeled_fasta, records_to_arrays
from .features import extract_features
from .metrics import classification_metrics
from .splits import make_outer_splits


METRICS = ("accuracy", "sensitivity", "specificity", "mcc", "f1", "auc_roc")
ALPHAS = (0.001, 0.005, 0.01, 0.02)


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    dimension: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[float, list[dict[str, object]]]:
    """Tune alpha while fitting PCA only once per inner fold."""
    if groups is None:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        splits = splitter.split(x, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        splits = splitter.split(x, y, groups=groups)
    predictions = {alpha: np.full(len(y), np.nan) for alpha in ALPHAS}
    for inner_fold, (train, valid) in enumerate(splits):
        transformer = SRCTransformer(mode=f"pca{dimension}", seed=seed + inner_fold).fit(
            x[train], y[train]
        )
        train_z = transformer.transform(x[train])
        valid_z = transformer.transform(x[valid])
        for alpha in ALPHAS:
            predictions[alpha][valid] = src_predict_proba(
                train_z, y[train], valid_z, alpha
            )[0]
    rows = [
        {
            "mode": f"pca{dimension}",
            "alpha": alpha,
            "mcc": float(matthews_corrcoef(y, probability >= 0.5)),
        }
        for alpha, probability in predictions.items()
    ]
    best = max(rows, key=lambda row: (float(row["mcc"]), -ALPHAS.index(float(row["alpha"]))))
    return float(best["alpha"]), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/positive_negative.fasta")
    parser.add_argument("--output-dir", default="results/revised_pca_dimension_ablation_5x5")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--seeds", nargs="+", type=int, default=[6, 17, 29, 41, 53])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--protocol", choices=("direct", "cluster"), default="direct")
    parser.add_argument("--cluster-file")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_labeled_fasta(args.data)
    sequences, y, identifiers = records_to_arrays(records)
    x = extract_features(sequences)
    repeat_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    for repeat, seed in enumerate(args.seeds):
        splits, outer_groups = make_outer_splits(
            y, identifiers, args.protocol, seed, args.folds, args.cluster_file
        )
        probabilities = {dimension: np.full(len(y), np.nan) for dimension in args.dimensions}
        for fold, (train, test) in enumerate(splits):
            fold_seed = seed + 100 * fold
            for dimension in args.dimensions:
                mode = f"pca{dimension}"
                train_groups = outer_groups[train] if args.protocol == "cluster" else None
                alpha, candidates = select_alpha(
                    x[train], y[train], dimension, fold_seed, groups=train_groups
                )
                transformer = SRCTransformer(mode=mode, seed=fold_seed).fit(x[train], y[train])
                train_z = transformer.transform(x[train])
                probability = src_predict_proba(
                    train_z, y[train], transformer.transform(x[test]), alpha
                )[0]
                probabilities[dimension][test] = probability
                fold_rows.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "fold": fold,
                        "dimension": dimension,
                        "selected_alpha": alpha,
                        "dictionary_atoms": len(train),
                        "overcompleteness_ratio": len(train) / dimension,
                        **classification_metrics(y[test], probability),
                    }
                )
                tuning_rows.extend(
                    {"repeat": repeat, "seed": seed, "fold": fold, "dimension": dimension,
                     "selected": row["alpha"] == alpha, **row}
                    for row in candidates
                )
        for dimension, probability in probabilities.items():
            if not np.isfinite(probability).all():
                raise AssertionError(f"Incomplete PCA{dimension} predictions, seed {seed}")
            repeat_rows.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "dimension": dimension,
                    **classification_metrics(y, probability),
                }
            )
        print(f"completed PCA ablation repeat {repeat + 1}/{len(args.seeds)} (seed={seed})", flush=True)

    repeat_frame = pd.DataFrame(repeat_rows)
    repeat_frame.to_csv(output / "repeat_metrics.csv", index=False)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(output / "inner_selection.csv", index=False)
    aggregate = repeat_frame.groupby("dimension")[list(METRICS)].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output / "aggregate_metrics.csv", index=False)
    mean_atoms = float(fold_frame["dictionary_atoms"].mean())
    structure = pd.DataFrame(
        [
            {"representation": "Original descriptors", "dimension": int(x.shape[1]),
             "mean_dictionary_atoms": mean_atoms, "overcompleteness_ratio": mean_atoms / x.shape[1]},
            *[
                {"representation": f"PCA{dimension}", "dimension": dimension,
                 "mean_dictionary_atoms": mean_atoms, "overcompleteness_ratio": mean_atoms / dimension}
                for dimension in args.dimensions
            ],
        ]
    )
    structure.to_csv(output / "dictionary_structure.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": (
                    "Repeated StratifiedKFold"
                    if args.protocol == "direct"
                    else "Repeated StratifiedGroupKFold with CD-HIT50 groups"
                ),
                "outer_folds": args.folds,
                "inner_folds_for_alpha": 3,
                "seeds": args.seeds,
                "dimensions": args.dimensions,
                "original_dimension": int(x.shape[1]),
                "dictionary_atoms_are_outer_training_peptides": True,
                "outer_test_used_for_selection": False,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(aggregate[["f1_mean", "f1_std", "mcc_mean", "mcc_std", "auc_roc_mean", "auc_roc_std"]])


if __name__ == "__main__":
    main()
