"""Repeated direct-CV comparison of PCA-SRC representation variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .classical import SRCTransformer, src_predict_proba, tune_src
from .data import read_labeled_fasta, records_to_arrays
from .deep_lse import DeepLSEConfig, encode_lse, select_and_refit_lse
from .features import extract_cksaap, extract_features, pearson_ranking, split_antidmppred_views
from .kernel_src import (
    AlignmentWeightedMultiKernelEmbedding,
    PolynomialCKSAAPEmbedding,
    l1_src_predict_proba,
    mp_src_predict_proba,
)
from .metrics import classification_metrics
from .splits import make_outer_splits


METRICS = ("accuracy", "sensitivity", "specificity", "mcc", "f1", "auc_roc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/positive_negative.fasta")
    parser.add_argument("--output-dir", default="results/revised_src_variants_160_5x5")
    parser.add_argument("--seeds", nargs="+", type=int, default=[6, 17, 29, 41, 53])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument(
        "--dimension",
        type=int,
        default=160,
        help="Common output dimension for every representation.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    dimension = int(args.dimension)
    if dimension <= 0:
        raise ValueError("--dimension must be positive")
    methods = (
        f"PCA{dimension}-SRC (proposed)",
        f"DeepLSE{dimension}-SRC",
        f"AE{dimension}-SRC",
        f"CKSAAP-KPCA{dimension}-MP",
        f"CKSAAP-KPCA{dimension}-L1-SRC",
        f"AWMKPCA{dimension}-L1-SRC",
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_labeled_fasta(args.data)
    sequences, y, identifiers = records_to_arrays(records)
    x = extract_features(sequences)
    cksaap = extract_cksaap(sequences, max_gap=8)
    views = split_antidmppred_views(x)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    base = DeepLSEConfig(input_dim=347, epochs=args.epochs, patience=args.patience)
    repeat_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    for repeat, seed in enumerate(args.seeds):
        splits, _ = make_outer_splits(y, identifiers, "direct", seed, args.folds)
        probabilities = {method: np.full(len(y), np.nan) for method in methods}
        fold_assignment = np.full(len(y), -1, dtype=int)
        for fold, (train, test) in enumerate(splits):
            fold_assignment[test] = fold
            fold_seed = seed + 100 * fold

            # All representations use the same output width in this controlled comparison.
            _, pca_alpha, pca_candidates = tune_src(
                x[train], y[train], fold_seed,
                modes=(f"pca{dimension}",), alphas=(0.001, 0.005, 0.01, 0.02),
            )
            pca = SRCTransformer(mode=f"pca{dimension}", seed=fold_seed).fit(x[train], y[train])
            probabilities[methods[0]][test] = src_predict_proba(
                pca.transform(x[train]), y[train], pca.transform(x[test]), pca_alpha
            )[0]
            selection_rows.extend(
                {"repeat": repeat, "seed": seed, "fold": fold, "method": methods[0],
                 "selected": row["alpha"] == pca_alpha, **row}
                for row in pca_candidates
            )

            # DeepLSE and reconstruction-only AE specifications are frozen before repeats.
            feature_indices = pearson_ranking(x[train], y[train])[:347]
            deep_model, deep_scaler, deep_meta, deep_rows = select_and_refit_lse(
                x[train][:, feature_indices], y[train], fold_seed, device,
                replace(base, latent_dim=dimension, classification_weight=1.0),
                latent_dims=(dimension,), alphas=(0.001, 0.005, 0.01, 0.02),
            )
            train_deep = encode_lse(deep_model, deep_scaler, x[train][:, feature_indices], device)
            test_deep = encode_lse(deep_model, deep_scaler, x[test][:, feature_indices], device)
            probabilities[methods[1]][test] = src_predict_proba(
                train_deep, y[train], test_deep, float(deep_meta["selected_alpha"])
            )[0]
            selection_rows.extend(
                {"repeat": repeat, "seed": seed, "fold": fold, "method": methods[1], **row}
                for row in deep_rows
            )

            ae_model, ae_scaler, ae_meta, ae_rows = select_and_refit_lse(
                x[train][:, feature_indices], y[train], fold_seed + 50_000, device,
                replace(base, latent_dim=dimension, classification_weight=0.0),
                latent_dims=(dimension,), alphas=(0.001, 0.005, 0.01, 0.02),
            )
            train_ae = encode_lse(ae_model, ae_scaler, x[train][:, feature_indices], device)
            test_ae = encode_lse(ae_model, ae_scaler, x[test][:, feature_indices], device)
            probabilities[methods[2]][test] = src_predict_proba(
                train_ae, y[train], test_ae, float(ae_meta["selected_alpha"])
            )[0]
            selection_rows.extend(
                {"repeat": repeat, "seed": seed, "fold": fold, "method": methods[2], **row}
                for row in ae_rows
            )

            # CKSAAP polynomial-kernel dictionary with two sparse solvers.
            kernel = PolynomialCKSAAPEmbedding(n_components=dimension, degree=2, seed=fold_seed).fit(
                cksaap[train], y[train]
            )
            train_kernel = kernel.transform(cksaap[train])
            test_kernel = kernel.transform(cksaap[test])
            probabilities[methods[3]][test] = mp_src_predict_proba(
                train_kernel, y[train], test_kernel, max_atoms=10
            )[0]
            probabilities[methods[4]][test] = l1_src_predict_proba(
                train_kernel, y[train], test_kernel, alpha=0.001
            )[0]

            train_views = {name: matrix[train] for name, matrix in views.items()}
            test_views = {name: matrix[test] for name, matrix in views.items()}
            awmk = AlignmentWeightedMultiKernelEmbedding(n_components=dimension, seed=fold_seed).fit(
                train_views, y[train]
            )
            probabilities[methods[5]][test] = l1_src_predict_proba(
                awmk.transform(train_views), y[train], awmk.transform(test_views), alpha=0.001
            )[0]
            weight_rows.extend(
                {"repeat": repeat, "seed": seed, "fold": fold, "view": name, "weight": weight}
                for name, weight in awmk.weight_map.items()
            )

            for method in methods:
                fold_rows.append(
                    {"repeat": repeat, "seed": seed, "fold": fold, "method": method,
                     **classification_metrics(y[test], probabilities[method][test])}
                )
            print(
                f"repeat {repeat + 1}/{len(args.seeds)} seed={seed}, fold {fold + 1}/{args.folds}",
                flush=True,
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
                    **{method: probabilities[method][index] for method in methods},
                }
            )

    repeat_frame = pd.DataFrame(repeat_rows)
    repeat_frame.to_csv(output / "repeat_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "inner_selection.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output / "kernel_view_weights.csv", index=False)
    aggregate = repeat_frame.groupby("method")[list(METRICS)].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output / "aggregate_metrics.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": "Repeated StratifiedKFold",
                "outer_folds": args.folds,
                "seeds": args.seeds,
                "device": device,
                "common_output_dimension": dimension,
                "pca_src": {"dimension": dimension, "alpha": "inner three-fold selected"},
                "deep_lse_src": {"input": "outer-training Pearson top 347", "dimension": dimension,
                                 "alpha": "inner holdout selected", "classification_weight": 1.0},
                "ae_src": {"input": "outer-training Pearson top 347", "dimension": dimension,
                           "alpha": "inner holdout selected", "classification_weight": 0.0},
                "cksaap_kpca": {"features": 3600, "dimension": dimension, "degree": 2,
                                "mp_atoms": 10, "l1_alpha": 0.001},
                "awmkpca": {"dimension": dimension, "l1_alpha": 0.001,
                            "weights": "outer-training centered kernel alignment"},
                "test_fold_used_for_training_or_selection": False,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(aggregate[["accuracy_mean", "accuracy_std", "mcc_mean", "mcc_std", "f1_mean", "f1_std", "auc_roc_mean", "auc_roc_std"]])


if __name__ == "__main__":
    main()
