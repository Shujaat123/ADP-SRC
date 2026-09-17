#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adp_src.classical import SRCTransformer, tune_src
from adp_src.data import read_labeled_fasta, records_to_arrays
from adp_src.features import extract_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the finalized PCA-SRC dictionary")
    parser.add_argument("--data", default="data/positive_negative.fasta")
    parser.add_argument("--output-dir", default="final_model/pca160_src")
    parser.add_argument("--dimension", type=int, default=160)
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_labeled_fasta(args.data)
    sequences, y, identifiers = records_to_arrays(records)
    x = extract_features(sequences)
    mode = f"pca{args.dimension}"
    _, alpha, scores = tune_src(
        x,
        y,
        seed=args.seed,
        modes=(mode,),
        alphas=(0.001, 0.005, 0.01, 0.02),
        inner_splits=5,
    )
    transformer = SRCTransformer(mode=mode, seed=args.seed).fit(x, y)
    dictionary = transformer.transform(x)
    np.savez_compressed(
        output / f"pca{args.dimension}_src_model.npz",
        dictionary=dictionary,
        labels=y,
        identifiers=np.asarray(identifiers),
        sequences=np.asarray(sequences),
        feature_indices=transformer.indices_,
        scaler_mean=transformer.scaler_.mean_,
        scaler_scale=transformer.scaler_.scale_,
        pca_mean=transformer.pca_.mean_,
        pca_components=transformer.pca_.components_,
        alpha=np.asarray([alpha]),
        seed=np.asarray([args.seed]),
    )
    metadata = {
        "model": f"PCA{args.dimension}-SRC",
        "dictionary_atoms": len(dictionary),
        "dictionary_dimension": dictionary.shape[1],
        "dictionary_shape": list(dictionary.shape),
        "overcomplete": dictionary.shape[0] > dictionary.shape[1],
        "class_atoms": {
            "negative": int(np.sum(y == 0)),
            "positive": int(np.sum(y == 1)),
        },
        "alpha": alpha,
        "alpha_selection": "5-fold StratifiedKFold on the complete development set",
        "alpha_candidates": [0.001, 0.005, 0.01, 0.02],
        "seed": args.seed,
        "performance_warning": (
            "The serialized dictionary is fitted on all 472 samples for future inference. "
            "Reported performance must come only from saved outer-fold OOF predictions."
        ),
        "inner_scores": scores,
    }
    (output / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "inner_scores"}, indent=2))


if __name__ == "__main__":
    main()
