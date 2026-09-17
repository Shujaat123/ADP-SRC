#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from adp_src.classical import src_predict_proba
from adp_src.features import AMINO_ACIDS, extract_features


def read_fasta(path: str | Path) -> tuple[list[str], list[str]]:
    identifiers: list[str] = []
    sequences: list[str] = []
    current: str | None = None
    pieces: list[str] = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    sequences.append("".join(pieces).upper())
                current = line[1:].split()[0]
                identifiers.append(current)
                pieces = []
            else:
                if current is None:
                    raise ValueError("Sequence encountered before FASTA header")
                pieces.append(line)
    if current is not None:
        sequences.append("".join(pieces).upper())
    if len(identifiers) != len(sequences) or not identifiers:
        raise ValueError("No complete FASTA records found")
    for identifier, sequence in zip(identifiers, sequences):
        invalid = sorted(set(sequence) - set(AMINO_ACIDS))
        if invalid or not 5 <= len(sequence) <= 50:
            raise ValueError(f"Invalid peptide >{identifier}: length={len(sequence)}, residues={invalid}")
    return identifiers, sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with a finalized PCA-SRC model")
    parser.add_argument("--model", default="final_model/pca160_src/pca160_src_model.npz")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = np.load(args.model)
    identifiers, sequences = read_fasta(args.input)
    x = extract_features(sequences)[:, model["feature_indices"]]
    scaled = (x - model["scaler_mean"]) / model["scaler_scale"]
    latent = (scaled - model["pca_mean"]) @ model["pca_components"].T
    latent = normalize(latent, norm="l2", axis=1)
    probability, residuals, coefficients = src_predict_proba(
        model["dictionary"],
        model["labels"],
        latent,
        float(model["alpha"][0]),
    )
    pd.DataFrame(
        {
            "identifier": identifiers,
            "sequence": sequences,
            "predicted_label": (probability >= 0.5).astype(int),
            "positive_probability": probability,
            "negative_residual": residuals[:, 0],
            "positive_residual": residuals[:, 1],
            "nonzero_atoms": np.count_nonzero(np.abs(coefficients) > 1e-10, axis=1),
        }
    ).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
