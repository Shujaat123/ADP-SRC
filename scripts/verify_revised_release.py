#!/usr/bin/env python3
"""Audit the PCA160-SRC release and extended ablation results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [6, 17, 29, 41, 53]
DIMENSIONS = [16, 32, 64, 96, 128, 160, 192, 224, 240]
EXPECTED_HASH = "9371de76e93319d8f12cf6ec6ce1ad3b565476cf2cd1cfdcef4422574b132edb"


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"Expected {expected:.16g}, found {actual:.16g}")


def one(frame: pd.DataFrame, column: str, value: object) -> pd.Series:
    rows = frame[frame[column] == value]
    if len(rows) != 1:
        raise AssertionError(f"Expected one {column}={value!r}, found {len(rows)}")
    return rows.iloc[0]


def verify_repeated_directory(path: Path, expected_repeats: int) -> None:
    aggregate = pd.read_csv(path / "aggregate_metrics.csv")
    repeat = pd.read_csv(path / "repeat_metrics.csv")
    fold = pd.read_csv(path / "fold_metrics.csv")
    if sorted(repeat["seed"].unique().tolist()) != SEEDS:
        raise AssertionError(f"Unexpected seeds in {path}")
    if len(repeat) != expected_repeats or len(fold) != expected_repeats * 5:
        raise AssertionError(f"Incomplete repeated evaluation in {path}")
    if not np.isfinite(aggregate.select_dtypes(include=[np.number]).to_numpy()).all():
        raise AssertionError(f"Non-finite aggregate metric in {path}")


def main() -> None:
    fasta = ROOT / "data/positive_negative.fasta"
    if hashlib.sha256(fasta.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise AssertionError("Historical benchmark checksum mismatch")

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if config["selected_model"] != "PCA160-SRC":
        raise AssertionError("PCA160-SRC is not the selected release model")
    selection = config["dimension_selection"]
    if selection["tested_dimensions"] != DIMENSIONS:
        raise AssertionError("Unexpected PCA ablation grid")
    if selection["literal_f1_maximum_dimension"] != 240 or selection["selected_dimension"] != 160:
        raise AssertionError("Dimension-selection record changed")

    direct_dir = ROOT / "results/revised_direct_comparators_5x5"
    variant_dir = ROOT / "results/revised_src_variants_160_5x5"
    ablation_dir = ROOT / "results/revised_pca_dimension_ablation_extended_5x5"
    cluster_dir = ROOT / "results/revised_pca160_cdhit50_5x5"
    ant_cluster_dir = ROOT / "results/stability_cluster_5x5"
    verify_repeated_directory(direct_dir, 25)
    verify_repeated_directory(variant_dir, 30)
    verify_repeated_directory(ablation_dir, 45)
    verify_repeated_directory(cluster_dir, 5)
    verify_repeated_directory(ant_cluster_dir, 15)

    pca = pd.read_csv(ablation_dir / "aggregate_metrics.csv")
    if pca["dimension"].tolist() != DIMENSIONS:
        raise AssertionError("Incomplete or unsorted PCA ablation")
    pca160 = one(pca, "dimension", 160)
    pca240 = one(pca, "dimension", 240)
    close(pca160["accuracy_mean"], 0.7906779661016949)
    close(pca160["mcc_mean"], 0.5818363895223703)
    close(pca160["f1_mean"], 0.7947013993868601)
    close(pca160["auc_roc_mean"], 0.881072249353634)
    if int(pca.loc[pca["f1_mean"].idxmax(), "dimension"]) != 240:
        raise AssertionError("PCA240 is no longer the literal mean-F1 maximum")
    if pca240["f1_mean"] - pca160["f1_mean"] > 0.005:
        raise AssertionError("PCA160 is no longer on the defined F1 plateau")
    if pca160["mcc_mean"] < pca["mcc_mean"].max() - 1e-12:
        raise AssertionError("PCA160 no longer has the highest mean MCC")
    if pca160["auc_roc_mean"] < pca["auc_roc_mean"].max() - 1e-12:
        raise AssertionError("PCA160 no longer has the highest mean AUC")
    if not pca160["f1_std"] < pca240["f1_std"]:
        raise AssertionError("PCA160 no longer has lower F1 variability than PCA240")

    direct = pd.read_csv(direct_dir / "aggregate_metrics.csv")
    rbf = one(direct, "method", "RBF-SVM")
    ant = one(direct, "method", "AntiDMPpred-RF (reimplemented)")
    close(rbf["accuracy_mean"], 0.775)
    close(ant["accuracy_mean"], 0.7300847457627119)
    if not (pca160["f1_mean"] > rbf["f1_mean"] > ant["f1_mean"]):
        raise AssertionError("Primary mean-F1 ordering changed")

    variants = pd.read_csv(variant_dir / "aggregate_metrics.csv")
    expected_methods = {
        "PCA160-SRC (proposed)",
        "DeepLSE160-SRC",
        "AE160-SRC",
        "CKSAAP-KPCA160-MP",
        "CKSAAP-KPCA160-L1-SRC",
        "AWMKPCA160-L1-SRC",
    }
    if set(variants["method"]) != expected_methods:
        raise AssertionError("Common-width SRC method set changed")
    variant_meta = json.loads((variant_dir / "metadata.json").read_text(encoding="utf-8"))
    if variant_meta["common_output_dimension"] != 160:
        raise AssertionError("SRC representation comparison is not common-width 160D")
    kernel = one(variants, "method", "CKSAAP-KPCA160-L1-SRC")
    awmk = one(variants, "method", "AWMKPCA160-L1-SRC")
    deep = one(variants, "method", "DeepLSE160-SRC")
    autoencoder = one(variants, "method", "AE160-SRC")
    close(kernel["accuracy_mean"], 0.7843220338983051)
    close(kernel["mcc_mean"], 0.5695645770770081)
    close(kernel["f1_mean"], 0.7785845014997493)
    close(kernel["auc_roc_mean"], 0.8810399310542947)
    close(awmk["f1_mean"], 0.7962248946033482)
    close(deep["mcc_mean"], 0.45383427298119106)
    close(autoencoder["mcc_mean"], 0.4815438966642217)
    if not pca160["accuracy_mean"] > variants["accuracy_mean"].drop(
        variants.index[variants["method"] == "PCA160-SRC (proposed)"]
    ).max():
        raise AssertionError("PCA160-SRC no longer has the highest common-width accuracy")
    if not pca160["mcc_mean"] > variants["mcc_mean"].drop(
        variants.index[variants["method"] == "PCA160-SRC (proposed)"]
    ).max():
        raise AssertionError("PCA160-SRC no longer has the highest common-width MCC")
    if abs(pca160["auc_roc_mean"] - kernel["auc_roc_mean"]) > 0.001:
        raise AssertionError("CKSAAP-KPCA160 no longer closely matches PCA160 AUC")
    if not awmk["f1_mean"] > pca160["f1_mean"]:
        raise AssertionError("Expected AWMKPCA160 marginal F1 lead changed")

    pca_cluster = one(pd.read_csv(cluster_dir / "aggregate_metrics.csv"), "dimension", 160)
    ant_cluster = one(
        pd.read_csv(ant_cluster_dir / "aggregate_metrics.csv"),
        "method", "AntiDMPpred-RF (reimplemented)",
    )
    close(pca_cluster["accuracy_mean"], 0.714406779661017)
    close(pca_cluster["mcc_mean"], 0.42890684093220577)
    close(pca_cluster["f1_mean"], 0.7158173146702868)
    close(pca_cluster["auc_roc_mean"], 0.7905953748922723)
    close(ant_cluster["accuracy_mean"], 0.6817796610169491)

    structure = pd.read_csv(ablation_dir / "dictionary_structure.csv")
    raw = one(structure, "representation", "Original descriptors")
    latent = one(structure, "representation", "PCA160")
    if not (raw["overcompleteness_ratio"] < 1 < latent["overcompleteness_ratio"]):
        raise AssertionError("PCA does not convert the outer-fold dictionary to overcomplete")
    close(latent["overcompleteness_ratio"], 2.36)

    model = np.load(ROOT / "final_model/pca160_src/pca160_src_model.npz")
    if model["dictionary"].shape != (472, 160):
        raise AssertionError(f"Unexpected deployment dictionary: {model['dictionary'].shape}")
    if np.bincount(model["labels"].astype(int)).tolist() != [236, 236]:
        raise AssertionError("Unexpected deployment class counts")
    if not np.allclose(np.linalg.norm(model["dictionary"], axis=1), 1.0, atol=1e-8):
        raise AssertionError("Deployment atoms are not unit normalized")

    required = [
        "README.md",
        "MODEL_CARD.md",
        "CITATION.cff",
        "config.json",
        "data/positive_negative.fasta",
        "data/positive_negative_cdhit50.fasta.clstr",
        "final_model/pca160_src/pca160_src_model.npz",
        "results/revised_direct_comparators_5x5/aggregate_metrics.csv",
        "results/revised_src_variants_160_5x5/aggregate_metrics.csv",
        "results/revised_pca_dimension_ablation_extended_5x5/aggregate_metrics.csv",
        "results/revised_pca160_cdhit50_5x5/aggregate_metrics.csv",
    ]
    for relative in required:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing release artifact: {relative}")

    prohibited_paths = [
        ROOT / "paper_revised",
        ROOT / "scripts/build_manuscript.sh",
        ROOT / "scripts/build_revised_manuscript.sh",
        ROOT / "scripts/build_revised_paper_outputs.py",
    ]
    for path in prohibited_paths:
        if path.exists():
            raise AssertionError(f"Private manuscript artifact present: {path.relative_to(ROOT)}")
    for pattern in ("*.tex", "*.bib"):
        matches = list(ROOT.rglob(pattern))
        if matches:
            raise AssertionError(
                f"Private manuscript source present: {matches[0].relative_to(ROOT)}"
            )

    print("ADP_PCA160_SRC_COMMON_WIDTH_160_VERIFIED")


if __name__ == "__main__":
    main()
