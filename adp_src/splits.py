from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def parse_cdhit_clusters(path: str | Path) -> dict[str, int]:
    cluster = -1
    mapping: dict[str, int] = {}
    member_pattern = re.compile(r">([^\.\s]+)")
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">Cluster"):
                cluster += 1
                continue
            match = member_pattern.search(line)
            if match:
                identifier = match.group(1)
                mapping[identifier] = cluster
    if not mapping:
        raise ValueError(f"No CD-HIT members found in {path}")
    return mapping


def make_outer_splits(
    y: np.ndarray,
    identifiers: list[str],
    protocol: str,
    seed: int,
    n_splits: int = 5,
    cluster_file: str | Path | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    y = np.asarray(y, dtype=np.int64)
    if protocol == "direct":
        groups = np.arange(len(y), dtype=np.int64)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y))
    elif protocol == "cluster":
        if cluster_file is None:
            raise ValueError("cluster protocol requires --cluster-file")
        mapping = parse_cdhit_clusters(cluster_file)
        missing = [identifier for identifier in identifiers if identifier not in mapping]
        if missing:
            raise ValueError(f"{len(missing)} identifiers missing from CD-HIT clusters")
        groups = np.asarray([mapping[identifier] for identifier in identifiers], dtype=np.int64)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y, groups=groups))
    else:
        raise ValueError("protocol must be 'direct' or 'cluster'")

    seen = np.zeros(len(y), dtype=np.int64)
    for fold, (train, test) in enumerate(splits):
        if set(groups[train]).intersection(groups[test]):
            raise AssertionError(f"Group leakage in fold {fold}")
        seen[test] += 1
    if not np.all(seen == 1):
        raise AssertionError("Every sample must occur in exactly one test fold")
    return splits, groups


def save_split_manifest(
    path: str | Path,
    identifiers: list[str],
    sequences: list[str],
    y: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    fold_id = np.full(len(y), -1, dtype=np.int64)
    for fold, (_, test) in enumerate(splits):
        fold_id[test] = fold
    pd.DataFrame(
        {
            "index": np.arange(len(y)),
            "identifier": identifiers,
            "sequence": sequences,
            "label": y,
            "cluster_id": groups,
            "test_fold": fold_id,
        }
    ).to_csv(path, index=False)
