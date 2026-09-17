"""Repeated direct-CV comparison of an initial PCA64 reference and conventional classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .classical import (
    SRCTransformer,
    fit_gradient_boosting,
    fit_logistic_regression,
    fit_rbf_svm,
    fit_reference_rf,
    src_predict_proba,
    tune_src,
)
from .data import read_labeled_fasta, records_to_arrays
from .features import extract_features
from .metrics import classification_metrics
from .splits import make_outer_splits


METHODS = (
    "PCA64-SRC (initial reference)",
    "AntiDMPpred-RF (reimplemented)",
    "RBF-SVM",
    "Gradient Boosting",
    "Logistic Regression",
)
METRICS = ("accuracy", "sensitivity", "specificity", "mcc", "f1", "auc_roc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/positive_negative.fasta")
    parser.add_argument("--output-dir", default="results/revised_direct_comparators_5x5")
    parser.add_argument("--seeds", nargs="+", type=int, default=[6, 17, 29, 41, 53])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_labeled_fasta(args.data)
    sequences, y, identifiers = records_to_arrays(records)
    x = extract_features(sequences)
    repeat_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    for repeat, seed in enumerate(args.seeds):
        splits, _ = make_outer_splits(y, identifiers, "direct", seed, args.folds)
        probabilities = {method: np.full(len(y), np.nan) for method in METHODS}
        fold_assignment = np.full(len(y), -1, dtype=int)
        for fold, (train, test) in enumerate(splits):
            fold_assignment[test] = fold
            fold_seed = seed + 100 * fold
            _, alpha, candidates = tune_src(
                x[train], y[train], fold_seed,
                modes=("pca64",), alphas=(0.001, 0.005, 0.01, 0.02),
            )
            transformer = SRCTransformer(mode="pca64", seed=fold_seed).fit(x[train], y[train])
            probabilities[METHODS[0]][test] = src_predict_proba(
                transformer.transform(x[train]), y[train], transformer.transform(x[test]), alpha
            )[0]
            probabilities[METHODS[1]][test] = fit_reference_rf(
                x[train], y[train], x[test], fold_seed
            )[0]
            probabilities[METHODS[2]][test] = fit_rbf_svm(
                x[train], y[train], x[test], fold_seed
            )
            probabilities[METHODS[3]][test] = fit_gradient_boosting(
                x[train], y[train], x[test], fold_seed
            )
            probabilities[METHODS[4]][test] = fit_logistic_regression(
                x[train], y[train], x[test], fold_seed
            )
            tuning_rows.extend(
                {"repeat": repeat, "seed": seed, "fold": fold, "selected": row["alpha"] == alpha, **row}
                for row in candidates
            )
            for method in METHODS:
                fold_rows.append(
                    {"repeat": repeat, "seed": seed, "fold": fold, "method": method,
                     **classification_metrics(y[test], probabilities[method][test])}
                )

        for method, probability in probabilities.items():
            if not np.isfinite(probability).all():
                raise AssertionError(f"Incomplete predictions: {method}, seed {seed}")
            repeat_rows.append(
                {"repeat": repeat, "seed": seed, "method": method,
                 **classification_metrics(y, probability)}
            )
        for index, identifier in enumerate(identifiers):
            prediction_rows.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "index": index,
                    "identifier": identifier,
                    "label": int(y[index]),
                    "test_fold": int(fold_assignment[index]),
                    **{method: probabilities[method][index] for method in METHODS},
                }
            )
        print(f"completed direct repeat {repeat + 1}/{len(args.seeds)} (seed={seed})", flush=True)

    repeat_frame = pd.DataFrame(repeat_rows)
    repeat_frame.to_csv(output / "repeat_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(output / "inner_selection.csv", index=False)
    aggregate = repeat_frame.groupby("method")[list(METRICS)].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output / "aggregate_metrics.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": "Repeated StratifiedKFold",
                "outer_folds": args.folds,
                "inner_folds_for_src_alpha": 3,
                "seeds": args.seeds,
                "feature_dimension": int(x.shape[1]),
                "anti_dmppred_values": "same-fold reimplementation only",
                "test_fold_used_for_training_or_selection": False,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(aggregate[["accuracy_mean", "accuracy_std", "mcc_mean", "mcc_std", "f1_mean", "f1_std", "auc_roc_mean", "auc_roc_std"]])


if __name__ == "__main__":
    main()
