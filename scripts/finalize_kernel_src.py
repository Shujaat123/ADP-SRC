#!/usr/bin/env python3
"""Fit and serialize the selected CKSAAP-KPCA-L1-SRC model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import normalize

from adp_src.data import read_labeled_fasta, records_to_arrays
from adp_src.features import extract_cksaap
from adp_src.kernel_evaluation import _score_candidates
from adp_src.kernel_src import PolynomialCKSAAPEmbedding
from adp_src.splits import parse_cdhit_clusters


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/positive_negative.fasta")
    parser.add_argument(
        "--cluster-file", default="data/positive_negative_cdhit50.fasta.clstr"
    )
    parser.add_argument("--output-dir", default="final_model/kernel_src")
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_labeled_fasta(args.data)
    sequences, y, identifiers = records_to_arrays(records)
    raw = extract_cksaap(sequences, max_gap=8)
    cluster_map = parse_cdhit_clusters(args.cluster_file)
    groups = np.asarray([cluster_map[identifier] for identifier in identifiers])

    selected_by_solver, candidates = _score_candidates(
        "CKSAAP", raw, y, args.seed, groups
    )
    selected = selected_by_solver["L1"]
    dimension = int(selected["dimension"])
    alpha = float(selected["solver_value"])
    embedding = PolynomialCKSAAPEmbedding(
        n_components=dimension, degree=2, seed=args.seed
    ).fit(raw, y)
    dictionary = normalize(embedding.transform(raw), axis=1)
    if len(dictionary) <= dictionary.shape[1]:
        raise AssertionError("Final dictionary is not overcomplete")

    model = {
        "model_name": "CKSAAP-KPCA-L1-SRC",
        "embedding": embedding,
        "dictionary": dictionary,
        "dictionary_labels": y,
        "dictionary_identifiers": identifiers,
        "max_gap": 8,
        "alpha": alpha,
        "decision_threshold": 0.5,
    }
    model_path = output / "cksaap_kpca_l1_src.joblib"
    joblib.dump(model, model_path, compress=3)
    metadata = {
        "model_name": model["model_name"],
        "selection_protocol": "3-fold StratifiedGroupKFold using CD-HIT50 groups; MCC then AUC",
        "selected_dimension": dimension,
        "selected_alpha": alpha,
        "dictionary_atoms": len(dictionary),
        "dictionary_dimension": int(dictionary.shape[1]),
        "overcompleteness_ratio": float(len(dictionary) / dictionary.shape[1]),
        "dataset_sha256": sha256(args.data),
        "model_sha256": sha256(model_path),
        "historical_negative_warning": "Class 0 comprises AVPdb-derived historical benchmark controls, not assay-confirmed universal negatives.",
        "candidates": candidates,
    }
    (output / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
