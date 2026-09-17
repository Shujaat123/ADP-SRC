from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.stats import binomtest
from sklearn.metrics import roc_curve

from .classical import (
    SRCTransformer,
    fit_rbf_svm,
    fit_reference_rf,
    src_predict_proba,
    tune_src,
)
from .data import read_labeled_fasta, records_to_arrays
from .deep_lse import (
    DeepLSEConfig,
    encode_lse,
    fit_latent_heads,
    predict_lse_classifier,
    select_and_refit_lse,
)
from .features import extract_features, pearson_ranking
from .metrics import bootstrap_interval, classification_metrics
from .splits import make_outer_splits, save_split_manifest


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plot_roc(y: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    plt.figure(figsize=(8.0, 6.0))
    for method, probability in predictions.items():
        fpr, tpr, _ = roc_curve(y, probability)
        auc = classification_metrics(y, probability)["auc_roc"]
        plt.plot(fpr, tpr, linewidth=1.8, label=f"{method} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("AntiDMPpred benchmark: out-of-fold ROC")
    plt.legend(loc="lower right", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_metrics(summary: pd.DataFrame, path: Path) -> None:
    metrics = ["accuracy", "auc_roc", "mcc", "f1"]
    positions = np.arange(len(summary))
    width = 0.19
    plt.figure(figsize=(11.0, 6.0))
    for offset, metric in enumerate(metrics):
        plt.bar(
            positions + (offset - 1.5) * width,
            summary[metric],
            width,
            label=metric.upper(),
        )
    plt.xticks(positions, summary["method"], rotation=22, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("SRC and DeepLSE representation comparison")
    plt.legend(ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _mcnemar(y: np.ndarray, predictions: dict[str, np.ndarray], reference: str) -> pd.DataFrame:
    reference_correct = (predictions[reference] >= 0.5) == y
    rows: list[dict[str, object]] = []
    for method, probability in predictions.items():
        if method == reference:
            continue
        current_correct = (probability >= 0.5) == y
        reference_only = int(np.sum(reference_correct & ~current_correct))
        current_only = int(np.sum(~reference_correct & current_correct))
        discordant = reference_only + current_only
        p_value = (
            float(binomtest(min(reference_only, current_only), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        rows.append(
            {
                "reference": reference,
                "method": method,
                "reference_only_correct": reference_only,
                "method_only_correct": current_only,
                "exact_mcnemar_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    data_path = Path(args.data).resolve()
    records = read_labeled_fasta(data_path)
    sequences, y, identifiers = records_to_arrays(records)
    if len(y) != 472 or int(y.sum()) != 236:
        raise AssertionError("Expected the balanced 472-peptide AntiDMPpred benchmark")
    x = extract_features(sequences)
    splits, groups = make_outer_splits(
        y, identifiers, args.protocol, args.seed, args.folds, args.cluster_file
    )
    save_split_manifest(output / "split_manifest.csv", identifiers, sequences, y, groups, splits)

    method_names = [
        "AntiDMPpred-RF (reimplemented)",
        "RBF-SVM (520 descriptors)",
        "PCA64-SRC",
        "DeepLSE-CLS",
        "DeepLSE-SVM",
        "DeepLSE-RF",
        "DeepLSE-SRC",
        "AE-LSE-SRC (no class loss)",
    ]
    predictions = {method: np.full(len(y), np.nan) for method in method_names}
    fold_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    base_config = DeepLSEConfig(epochs=args.epochs, patience=args.patience)

    for fold, (train, test) in enumerate(splits):
        print(f"Fold {fold + 1}/{len(splits)}: train={len(train)}, test={len(test)}", flush=True)
        fold_seed = args.seed + 100 * fold
        train_groups = groups[train] if args.protocol == "cluster" else None
        if args.lse_features == "all":
            lse_indices = np.arange(x.shape[1])
        else:
            feature_count = int(args.lse_features.replace("pearson", ""))
            lse_indices = pearson_ranking(x[train], y[train])[:feature_count]
        fold_lse_config = replace(base_config, input_dim=len(lse_indices))

        start = time.perf_counter()
        probability, indices = fit_reference_rf(x[train], y[train], x[test], fold_seed)
        predictions[method_names[0]][test] = probability
        timing_rows.append({"fold": fold, "method": method_names[0], "seconds": time.perf_counter() - start})

        start = time.perf_counter()
        predictions[method_names[1]][test] = fit_rbf_svm(x[train], y[train], x[test], fold_seed)
        timing_rows.append({"fold": fold, "method": method_names[1], "seconds": time.perf_counter() - start})

        start = time.perf_counter()
        mode, alpha, scores = tune_src(
            x[train],
            y[train],
            fold_seed,
            modes=("pca64",),
            alphas=(0.001, 0.005, 0.01, 0.02),
            groups=train_groups,
        )
        transformer = SRCTransformer(mode=mode, seed=fold_seed).fit(x[train], y[train])
        train_pca = transformer.transform(x[train])
        test_pca = transformer.transform(x[test])
        probability, residuals, coefficients = src_predict_proba(
            train_pca, y[train], test_pca, alpha
        )
        predictions[method_names[2]][test] = probability
        timing_rows.append({"fold": fold, "method": method_names[2], "seconds": time.perf_counter() - start})
        tuning_rows.extend(
            {"fold": fold, "method": method_names[2], **row} for row in scores
        )
        pd.DataFrame(
            {
                "index": test,
                "residual_negative": residuals[:, 0],
                "residual_positive": residuals[:, 1],
                "nonzero_coefficients": np.count_nonzero(np.abs(coefficients) > 1e-10, axis=1),
            }
        ).to_csv(output / f"pca_src_diagnostics_fold{fold}.csv", index=False)

        start = time.perf_counter()
        joint_model, joint_scaler, joint_meta, joint_candidates = select_and_refit_lse(
            x[train][:, lse_indices],
            y[train],
            fold_seed,
            device,
            fold_lse_config,
            groups=train_groups,
        )
        train_z = encode_lse(joint_model, joint_scaler, x[train][:, lse_indices], device)
        test_z = encode_lse(joint_model, joint_scaler, x[test][:, lse_indices], device)
        head_predictions, residuals, coefficients = fit_latent_heads(
            train_z,
            y[train],
            test_z,
            float(joint_meta["selected_alpha"]),
            fold_seed,
        )
        predictions["DeepLSE-CLS"][test] = predict_lse_classifier(
            joint_model, joint_scaler, x[test][:, lse_indices], device
        )
        for method, probability in head_predictions.items():
            predictions[method][test] = probability
        elapsed = time.perf_counter() - start
        for method in ("DeepLSE-CLS", "DeepLSE-SVM", "DeepLSE-RF", "DeepLSE-SRC"):
            timing_rows.append({"fold": fold, "method": method, "seconds": elapsed})
        tuning_rows.extend(
            {"fold": fold, "method": "DeepLSE-SRC", **row} for row in joint_candidates
        )
        pd.DataFrame(
            {
                "index": test,
                "residual_negative": residuals[:, 0],
                "residual_positive": residuals[:, 1],
                "nonzero_coefficients": np.count_nonzero(np.abs(coefficients) > 1e-10, axis=1),
            }
        ).to_csv(output / f"deeplse_src_diagnostics_fold{fold}.csv", index=False)
        torch.save(
            {
                "state_dict": joint_model.state_dict(),
                "metadata": joint_meta,
                "scaler_mean": joint_scaler.mean_,
                "scaler_scale": joint_scaler.scale_,
            },
            output / "models" / f"deeplse_fold{fold}.pt",
        )

        start = time.perf_counter()
        ae_config = replace(fold_lse_config, classification_weight=0.0)
        ae_model, ae_scaler, ae_meta, ae_candidates = select_and_refit_lse(
            x[train][:, lse_indices],
            y[train],
            fold_seed + 50_000,
            device,
            ae_config,
            groups=train_groups,
        )
        train_ae = encode_lse(ae_model, ae_scaler, x[train][:, lse_indices], device)
        test_ae = encode_lse(ae_model, ae_scaler, x[test][:, lse_indices], device)
        probability, residuals, coefficients = src_predict_proba(
            train_ae, y[train], test_ae, float(ae_meta["selected_alpha"])
        )
        predictions[method_names[7]][test] = probability
        timing_rows.append({"fold": fold, "method": method_names[7], "seconds": time.perf_counter() - start})
        tuning_rows.extend(
            {"fold": fold, "method": method_names[7], **row} for row in ae_candidates
        )
        torch.save(
            {
                "state_dict": ae_model.state_dict(),
                "metadata": ae_meta,
                "scaler_mean": ae_scaler.mean_,
                "scaler_scale": ae_scaler.scale_,
            },
            output / "models" / f"ae_lse_fold{fold}.pt",
        )

        for method, probability in predictions.items():
            if np.isfinite(probability[test]).all():
                fold_rows.append(
                    {"fold": fold, "method": method, **classification_metrics(y[test], probability[test])}
                )
        print(
            "  selected joint latent/alpha/epochs="
            f"{joint_meta['latent_dim']}/{joint_meta['selected_alpha']}/{joint_meta['selected_epochs']}; "
            "AE="
            f"{ae_meta['latent_dim']}/{ae_meta['selected_alpha']}/{ae_meta['selected_epochs']}",
            flush=True,
        )

    for method, probability in predictions.items():
        if not np.isfinite(probability).all():
            raise AssertionError(f"Missing OOF predictions for {method}")
    pd.DataFrame(
        {"identifier": identifiers, "sequence": sequences, "label": y, **predictions}
    ).to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(output / "inner_selection.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output / "runtime_by_fold.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for method, probability in predictions.items():
        metrics = classification_metrics(y, probability)
        acc_low, acc_high = bootstrap_interval(y, probability, "accuracy")
        auc_low, auc_high = bootstrap_interval(y, probability, "auc_roc")
        summary_rows.append(
            {
                "method": method,
                "source": "this study",
                "protocol": f"{args.protocol} stratified {args.folds}-fold OOF; seed {args.seed}",
                **metrics,
                "accuracy_ci95_low": acc_low,
                "accuracy_ci95_high": acc_high,
                "auc_ci95_low": auc_low,
                "auc_ci95_high": auc_high,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mcc", "auc_roc"], ascending=False
    )
    summary.to_csv(output / "summary_metrics.csv", index=False)
    summary.to_csv(output / "model_comparison.csv", index=False)
    _mcnemar(y, predictions, "DeepLSE-SRC").to_csv(
        output / "mcnemar_vs_deeplse_src.csv", index=False
    )
    _mcnemar(y, predictions, "PCA64-SRC").to_csv(
        output / "mcnemar_vs_pca64_src.csv", index=False
    )
    _plot_roc(y, predictions, output / "roc_curves.png")
    _plot_metrics(summary, output / "metric_comparison.png")

    metadata = {
        "dataset": str(data_path),
        "dataset_sha256": sha256(data_path),
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "protocol": args.protocol,
        "cluster_file": args.cluster_file,
        "splitter": (
            "StratifiedKFold(shuffle=True)"
            if args.protocol == "direct"
            else "StratifiedGroupKFold(shuffle=True), groups=CD-HIT50 clusters"
        ),
        "outer_folds": args.folds,
        "seed": args.seed,
        "deep_lse_config": asdict(base_config),
        "deep_lse_features": args.lse_features,
        "deep_lse_effective_input_dim": (
            520 if args.lse_features == "all" else int(args.lse_features.replace("pearson", ""))
        ),
        "deep_lse_feature_selection_scope": "outer-training fold only",
        "inner_latent_dimensions": [32, 64, 128],
        "inner_src_alphas": [0.001, 0.005, 0.01, 0.02],
        "outer_test_used_for_model_selection": False,
        "dictionary_rule": "all outer-training latent vectors are atoms; no outer-test atom",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "device": device,
        "fidelity_note": (
            "AntiDMPpred-RF values in this study are from our same-fold reimplementation. "
            "The original PseAA_value.mat was unavailable, so exact software replication is not claimed."
        ),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nFinal OOF metrics")
    print(
        summary[
            ["method", "accuracy", "sensitivity", "specificity", "mcc", "f1", "auc_roc"]
        ].to_string(index=False)
    )
    print(f"\nSaved to {output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="SRC and DeepLSE AntiDMPpred study")
    result.add_argument("--data", default="data/positive_negative.fasta")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--protocol", choices=("direct", "cluster"), default="direct")
    result.add_argument("--cluster-file", default=None)
    result.add_argument("--folds", type=int, default=5)
    result.add_argument("--seed", type=int, default=6)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--patience", type=int, default=14)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument(
        "--lse-features",
        choices=("all", "pearson128", "pearson347"),
        default="pearson347",
    )
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
