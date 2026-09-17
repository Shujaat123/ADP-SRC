from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .classical import (
    SRCTransformer,
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
    "PCA64-SRC",
    "RBF-SVM (520 descriptors)",
    "AntiDMPpred-RF (reimplemented)",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated stratified stability analysis")
    parser.add_argument("--data", default="data/positive_negative.fasta")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=("direct", "cluster"), required=True)
    parser.add_argument("--cluster-file")
    parser.add_argument("--seeds", nargs="+", type=int, default=[6, 17, 29, 41, 53])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_labeled_fasta(args.data)
    _, y, identifiers = records_to_arrays(records)
    x = extract_features([record.sequence for record in records])
    repeat_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for repeat, seed in enumerate(args.seeds):
        splits, groups = make_outer_splits(
            y, identifiers, args.protocol, seed, args.folds, args.cluster_file
        )
        predictions = {method: np.full(len(y), np.nan) for method in METHODS}
        for fold, (train, test) in enumerate(splits):
            train_groups = groups[train] if args.protocol == "cluster" else None
            fold_seed = seed + 100 * fold
            mode, alpha, _ = tune_src(
                x[train],
                y[train],
                fold_seed,
                modes=("pca64",),
                alphas=(0.001, 0.005, 0.01, 0.02),
                groups=train_groups,
            )
            transformer = SRCTransformer(mode=mode, seed=fold_seed).fit(x[train], y[train])
            probability, _, _ = src_predict_proba(
                transformer.transform(x[train]),
                y[train],
                transformer.transform(x[test]),
                alpha,
            )
            predictions[METHODS[0]][test] = probability
            predictions[METHODS[1]][test] = fit_rbf_svm(
                x[train], y[train], x[test], fold_seed
            )
            predictions[METHODS[2]][test] = fit_reference_rf(
                x[train], y[train], x[test], fold_seed
            )[0]
            for method in METHODS:
                fold_rows.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "fold": fold,
                        "method": method,
                        **classification_metrics(y[test], predictions[method][test]),
                    }
                )
        for method, probability in predictions.items():
            repeat_rows.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "method": method,
                    **classification_metrics(y, probability),
                }
            )
        print(f"completed repeat {repeat + 1}/{len(args.seeds)} (seed={seed})", flush=True)

    repeat_frame = pd.DataFrame(repeat_rows)
    fold_frame = pd.DataFrame(fold_rows)
    repeat_frame.to_csv(output / "repeat_metrics.csv", index=False)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    metrics = ["accuracy", "sensitivity", "specificity", "mcc", "f1", "auc_roc"]
    aggregate = repeat_frame.groupby("method")[metrics].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output / "aggregate_metrics.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": args.protocol,
                "splitter": (
                    "Repeated StratifiedKFold"
                    if args.protocol == "direct"
                    else "Repeated StratifiedGroupKFold with CD-HIT50 groups"
                ),
                "folds": args.folds,
                "seeds": args.seeds,
                "interpretation": (
                    "Repeated CV is a stability analysis. Repeats share observations and are not "
                    "independent population experiments."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(aggregate[["accuracy_mean", "accuracy_std", "mcc_mean", "mcc_std", "auc_roc_mean", "auc_roc_std"]])


if __name__ == "__main__":
    main()
