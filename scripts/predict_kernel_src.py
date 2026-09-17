#!/usr/bin/env python3
"""Predict candidate peptides with the serialized CKSAAP-KPCA-L1-SRC model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from adp_src.data import AMINO_ACIDS
from adp_src.features import extract_cksaap
from adp_src.kernel_src import l1_src_predict_proba


def read_sequences(path: str | Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    header = None
    sequence: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                rows.append((header, "".join(sequence)))
            header, sequence = line[1:].split()[0], []
        else:
            sequence.append(line.upper())
    if header is not None:
        rows.append((header, "".join(sequence)))
    if not rows:
        raise ValueError("No FASTA sequences found")
    for identifier, peptide in rows:
        invalid = sorted(set(peptide) - set(AMINO_ACIDS))
        if invalid:
            raise ValueError(f"{identifier}: invalid residues {invalid}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    rows = read_sequences(args.input)
    raw = extract_cksaap([sequence for _, sequence in rows], max_gap=model["max_gap"])
    latent = model["embedding"].transform(raw)
    probability, residuals, code = l1_src_predict_proba(
        model["dictionary"],
        model["dictionary_labels"],
        latent,
        model["alpha"],
    )
    frame = pd.DataFrame(
        {
            "identifier": [identifier for identifier, _ in rows],
            "sequence": [sequence for _, sequence in rows],
            "probability_historical_ADP_class": probability,
            "predicted_class": (probability >= model["decision_threshold"]).astype(int),
            "residual_historical_negative": residuals[:, 0],
            "residual_ADP": residuals[:, 1],
            "nonzero_coefficients": (abs(code) > 1e-10).sum(axis=1),
        }
    )
    frame.to_csv(args.output, index=False)
    counts = frame["predicted_class"].value_counts().sort_index().to_dict()
    print(f"Wrote {len(frame)} predictions to {Path(args.output).resolve()}")
    print(f"Predicted class counts: {counts}")


if __name__ == "__main__":
    main()
