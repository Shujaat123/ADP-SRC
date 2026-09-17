#!/usr/bin/env python3
"""Merge the verified initial and per-seed extended PCA ablation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("accuracy", "sensitivity", "specificity", "mcc", "f1", "auc_roc")
SEEDS = (6, 17, 29, 41, 53)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-dir", default="results/revised_pca_dimension_ablation_5x5"
    )
    parser.add_argument(
        "--extension-template", default="results/revised_pca_dimension_extension_seed{seed}"
    )
    parser.add_argument(
        "--output-dir", default="results/revised_pca_dimension_ablation_extended_5x5"
    )
    args = parser.parse_args()

    initial = Path(args.initial_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    filenames = ("repeat_metrics.csv", "fold_metrics.csv", "inner_selection.csv")
    merged: dict[str, pd.DataFrame] = {}

    for filename in filenames:
        frames = [pd.read_csv(initial / filename)]
        for repeat, seed in enumerate(SEEDS):
            frame = pd.read_csv(Path(args.extension_template.format(seed=seed)) / filename)
            frame["repeat"] = repeat
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        identity = ["seed", "dimension"]
        if "fold" in combined:
            identity.append("fold")
        if "alpha" in combined:
            identity.append("alpha")
        if combined.duplicated(identity).any():
            raise AssertionError(f"Duplicate rows after merging {filename}")
        combined = combined.sort_values(identity).reset_index(drop=True)
        combined.to_csv(output / filename, index=False)
        merged[filename] = combined

    repeat = merged["repeat_metrics.csv"]
    expected_dimensions = [16, 32, 64, 96, 128, 160, 192, 224, 240]
    if sorted(repeat["seed"].unique()) != list(SEEDS):
        raise AssertionError("Unexpected seed set")
    if sorted(repeat["dimension"].unique()) != expected_dimensions:
        raise AssertionError("Unexpected PCA dimension set")
    if len(repeat) != len(SEEDS) * len(expected_dimensions):
        raise AssertionError("Incomplete repeated ablation")

    aggregate = repeat.groupby("dimension")[list(METRICS)].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)

    fold = merged["fold_metrics.csv"]
    mean_atoms = float(fold["dictionary_atoms"].mean())
    structure = pd.DataFrame(
        [
            {
                "representation": "Original descriptors",
                "dimension": 520,
                "mean_dictionary_atoms": mean_atoms,
                "overcompleteness_ratio": mean_atoms / 520,
            },
            *[
                {
                    "representation": f"PCA{dimension}",
                    "dimension": dimension,
                    "mean_dictionary_atoms": mean_atoms,
                    "overcompleteness_ratio": mean_atoms / dimension,
                }
                for dimension in expected_dimensions
            ],
        ]
    )
    structure.to_csv(output / "dictionary_structure.csv", index=False)

    best = aggregate.loc[aggregate["f1_mean"].idxmax()]
    metadata = {
        "protocol": "Repeated StratifiedKFold",
        "outer_folds": 5,
        "inner_folds_for_alpha": 3,
        "seeds": list(SEEDS),
        "dimensions": expected_dimensions,
        "original_dimension": 520,
        "dictionary_atoms_are_outer_training_peptides": True,
        "outer_test_used_for_selection": False,
        "initial_dimensions_reused_from_verified_run": [16, 32, 64, 128],
        "newly_executed_dimensions": [96, 160, 192, 224, 240],
        "selected_dimension_by_mean_f1": int(best["dimension"]),
        "selected_mean_f1": float(best["f1_mean"]),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        aggregate[
            ["dimension", "accuracy_mean", "mcc_mean", "f1_mean", "f1_std", "auc_roc_mean"]
        ].to_string(index=False)
    )
    print(f"selected_dimension={int(best['dimension'])}")


if __name__ == "__main__":
    main()
