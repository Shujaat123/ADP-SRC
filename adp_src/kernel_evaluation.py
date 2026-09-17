"""Nested evaluation of kernel sparse-representation models on AntiDMPpred."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import normalize

from .data import read_labeled_fasta, records_to_arrays
from .features import extract_cksaap, extract_features, split_antidmppred_views
from .kernel_src import (
    AlignmentWeightedMultiKernelEmbedding,
    PolynomialCKSAAPEmbedding,
    l1_src_predict_proba,
    mp_src_predict_proba,
)
from .metrics import bootstrap_interval, classification_metrics
from .splits import make_outer_splits, save_split_manifest


METHODS = (
    "CKSAAP-KPCA-MP",
    "CKSAAP-KPCA-L1-SRC",
    "AWMKPCA-MP-SRC",
    "AWMKPCA-L1-SRC",
)
DIMENSIONS = (16, 32, 64)
MP_ATOMS = (5, 10, 20)
L1_ALPHAS = (0.001, 0.005, 0.01)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _splitter(y: np.ndarray, groups: np.ndarray | None, seed: int):
    if groups is None:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y))
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups=groups))


def _score_candidates(
    representation: str,
    raw: np.ndarray | dict[str, np.ndarray],
    y: np.ndarray,
    seed: int,
    groups: np.ndarray | None,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    """Select latent size and sparse solver setting by inner 3-fold MCC/AUC."""
    candidate_predictions: dict[tuple[str, int, float], np.ndarray] = {}
    for solver in ("MP", "L1"):
        values = MP_ATOMS if solver == "MP" else L1_ALPHAS
        for dimension in DIMENSIONS:
            for value in values:
                candidate_predictions[(solver, dimension, float(value))] = np.full(len(y), np.nan)

    for inner_fold, (train, valid) in enumerate(_splitter(y, groups, seed)):
        if representation == "CKSAAP":
            embedding = PolynomialCKSAAPEmbedding(
                n_components=max(DIMENSIONS), degree=2, seed=seed + inner_fold
            ).fit(raw[train], y[train])
            train_z = embedding.transform(raw[train])
            valid_z = embedding.transform(raw[valid])
        elif representation == "AWMK":
            train_views = {name: matrix[train] for name, matrix in raw.items()}
            valid_views = {name: matrix[valid] for name, matrix in raw.items()}
            embedding = AlignmentWeightedMultiKernelEmbedding(
                n_components=max(DIMENSIONS), seed=seed + inner_fold
            ).fit(train_views, y[train])
            train_z = embedding.transform(train_views)
            valid_z = embedding.transform(valid_views)
        else:
            raise ValueError(representation)

        for dimension in DIMENSIONS:
            train_d = normalize(train_z[:, :dimension], axis=1)
            valid_d = normalize(valid_z[:, :dimension], axis=1)
            for atoms in MP_ATOMS:
                probability, _, _ = mp_src_predict_proba(
                    train_d, y[train], valid_d, max_atoms=atoms
                )
                candidate_predictions[("MP", dimension, float(atoms))][valid] = probability
            for alpha in L1_ALPHAS:
                probability, _, _ = l1_src_predict_proba(
                    train_d, y[train], valid_d, alpha=alpha
                )
                candidate_predictions[("L1", dimension, float(alpha))][valid] = probability

    rows: list[dict[str, float | int | str]] = []
    for (solver, dimension, value), probability in candidate_predictions.items():
        if not np.isfinite(probability).all():
            raise AssertionError("Incomplete inner predictions")
        prediction = probability >= 0.5
        rows.append(
            {
                "representation": representation,
                "solver": solver,
                "dimension": dimension,
                "solver_value": value,
                "mcc": float(matthews_corrcoef(y, prediction)),
                "auc_roc": float(roc_auc_score(y, probability)),
            }
        )

    best_rows: list[dict[str, float | int | str]] = []
    for solver in ("MP", "L1"):
        eligible = [row for row in rows if row["solver"] == solver]
        best_rows.append(
            max(
                eligible,
                key=lambda row: (
                    float(row["mcc"]),
                    float(row["auc_roc"]),
                    -DIMENSIONS.index(int(row["dimension"])),
                ),
            )
        )
    return {str(row["solver"]): row for row in best_rows}, rows


def _fit_outer(
    representation: str,
    raw: np.ndarray | dict[str, np.ndarray],
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    selected: dict[str, dict[str, float | int | str]],
    seed: int,
):
    max_dimension = max(int(row["dimension"]) for row in selected.values())
    if representation == "CKSAAP":
        embedding = PolynomialCKSAAPEmbedding(
            n_components=max_dimension, degree=2, seed=seed
        ).fit(raw[train], y[train])
        train_z = embedding.transform(raw[train])
        test_z = embedding.transform(raw[test])
        weights = {}
    else:
        train_views = {name: matrix[train] for name, matrix in raw.items()}
        test_views = {name: matrix[test] for name, matrix in raw.items()}
        embedding = AlignmentWeightedMultiKernelEmbedding(
            n_components=max_dimension, seed=seed
        ).fit(train_views, y[train])
        train_z = embedding.transform(train_views)
        test_z = embedding.transform(test_views)
        weights = embedding.weight_map

    outputs = {}
    for solver, row in selected.items():
        dimension = int(row["dimension"])
        train_d = normalize(train_z[:, :dimension], axis=1)
        test_d = normalize(test_z[:, :dimension], axis=1)
        if solver == "MP":
            probability, residuals, code = mp_src_predict_proba(
                train_d, y[train], test_d, max_atoms=int(row["solver_value"])
            )
        else:
            probability, residuals, code = l1_src_predict_proba(
                train_d, y[train], test_d, alpha=float(row["solver_value"])
            )
        outputs[solver] = (probability, residuals, code)
    return outputs, weights


def _plot_roc(y: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 5.3))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for color, (method, probability) in zip(colors, predictions.items()):
        fpr, tpr, _ = roc_curve(y, probability)
        ax.plot(fpr, tpr, lw=2.0, color=color, label=f"{method} ({roc_auc_score(y, probability):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#777777")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = read_labeled_fasta(args.data)
    sequences, y, identifiers = records_to_arrays(records)
    if len(y) != 472 or int(y.sum()) != 236:
        raise AssertionError("Expected the balanced 472-peptide AntiDMPpred benchmark")
    descriptors = extract_features(sequences)
    cksaap = extract_cksaap(sequences, max_gap=8)
    views = split_antidmppred_views(descriptors)
    splits, outer_groups = make_outer_splits(
        y, identifiers, args.protocol, args.seed, args.folds, args.cluster_file
    )
    save_split_manifest(
        output / "split_manifest.csv", identifiers, sequences, y, outer_groups, splits
    )

    predictions = {method: np.full(len(y), np.nan) for method in METHODS}
    selection_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []

    for fold, (train, test) in enumerate(splits):
        fold_seed = args.seed + 100 * fold
        train_groups = outer_groups[train] if args.protocol == "cluster" else None
        print(f"Fold {fold + 1}/{args.folds}: train={len(train)} test={len(test)}", flush=True)
        for representation, raw in (("CKSAAP", cksaap), ("AWMK", views)):
            start = time.perf_counter()
            selected, candidates = _score_candidates(
                representation,
                raw[train] if representation == "CKSAAP" else {k: v[train] for k, v in raw.items()},
                y[train],
                fold_seed,
                train_groups,
            )
            outputs, weights = _fit_outer(
                representation, raw, y, train, test, selected, fold_seed
            )
            elapsed = time.perf_counter() - start
            for row in candidates:
                selection_rows.append({"outer_fold": fold, "selected": False, **row})
            for solver, selected_row in selected.items():
                for row in selection_rows[-len(candidates):]:
                    if (
                        row["solver"] == solver
                        and row["dimension"] == selected_row["dimension"]
                        and float(row["solver_value"]) == float(selected_row["solver_value"])
                    ):
                        row["selected"] = True
                method = (
                    f"CKSAAP-KPCA-{solver}" if representation == "CKSAAP" and solver == "MP"
                    else "CKSAAP-KPCA-L1-SRC" if representation == "CKSAAP"
                    else f"AWMKPCA-{solver}-SRC"
                )
                probability, residuals, code = outputs[solver]
                predictions[method][test] = probability
                timing_rows.append(
                    {"fold": fold, "method": method, "seconds_shared_representation_and_tuning": elapsed}
                )
                metrics = classification_metrics(y[test], probability)
                fold_rows.append({"fold": fold, "method": method, **metrics})
                for position, sample_index in enumerate(test):
                    diagnostic_rows.append(
                        {
                            "fold": fold,
                            "method": method,
                            "index": int(sample_index),
                            "residual_negative": residuals[position, 0],
                            "residual_positive": residuals[position, 1],
                            "nonzero_coefficients": int(np.count_nonzero(np.abs(code[position]) > 1e-10)),
                        }
                    )
            for name, weight in weights.items():
                weight_rows.append({"fold": fold, "view": name, "weight": weight})
            print(
                f"  {representation}: MP d={selected['MP']['dimension']} k={selected['MP']['solver_value']}; "
                f"L1 d={selected['L1']['dimension']} alpha={selected['L1']['solver_value']}",
                flush=True,
            )

    for method, probability in predictions.items():
        if not np.isfinite(probability).all():
            raise AssertionError(f"Missing OOF predictions for {method}")
    pd.DataFrame(
        {"identifier": identifiers, "sequence": sequences, "label": y, **predictions}
    ).to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "inner_selection.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output / "kernel_view_weights.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(output / "src_diagnostics.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output / "runtime_by_fold.csv", index=False)

    summary_rows = []
    for method, probability in predictions.items():
        metrics = classification_metrics(y, probability)
        acc_low, acc_high = bootstrap_interval(y, probability, "accuracy", seed=2026)
        auc_low, auc_high = bootstrap_interval(y, probability, "auc_roc", seed=2027)
        summary_rows.append(
            {
                "method": method,
                "protocol": args.protocol,
                **metrics,
                "accuracy_ci95_low": acc_low,
                "accuracy_ci95_high": acc_high,
                "auc_ci95_low": auc_low,
                "auc_ci95_high": auc_high,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["mcc", "auc_roc"], ascending=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    _plot_roc(y, predictions, output / "kernel_roc.pdf")
    _plot_roc(y, predictions, output / "kernel_roc.png")

    metadata = {
        "dataset": str(Path(args.data).resolve()),
        "dataset_sha256": sha256(args.data),
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "n_negative_historical": int((y == 0).sum()),
        "protocol": args.protocol,
        "splitter": "StratifiedKFold" if args.protocol == "direct" else "StratifiedGroupKFold with CD-HIT50 groups",
        "outer_folds": args.folds,
        "inner_folds": 3,
        "seed": args.seed,
        "cksaap_dimension": int(cksaap.shape[1]),
        "kpca_dimensions": list(DIMENSIONS),
        "mp_atoms": list(MP_ATOMS),
        "l1_alphas": list(L1_ALPHAS),
        "polynomial_kernel": {"degree": 2, "gamma": "1/n_features", "coef0": 1.0},
        "multi_kernel": "RBF per view; median distance bandwidth; nonnegative centered-alignment weights",
        "dictionary": "all outer-training embeddings as atoms; overcomplete because atoms > latent dimension",
        "smote": "not used: benchmark is exactly balanced",
        "outer_test_used_for_fitting_or_selection": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary[["method", "accuracy", "mcc", "auc_roc"]].to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=("direct", "cluster"), default="direct")
    parser.add_argument("--cluster-file")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.protocol == "cluster" and not args.cluster_file:
        parser.error("--cluster-file is required for cluster protocol")
    run(args)


if __name__ == "__main__":
    main()
