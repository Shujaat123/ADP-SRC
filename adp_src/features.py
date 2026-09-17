"""The 520 sequence descriptors described in the AntiDMPpred paper.

The paper concatenates AAC (20), DPC (400), type-I PseAAC (25), and
CKSAAGP at gaps 0/1/2 (75). The authors did not publish the numeric contents
of ``PseAA_value.mat``; the conventional three Chou properties are used here,
matching the earlier audited Python reconstruction bundled with this project.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AA_GROUPS = ("GAVLMI", "FYW", "KRH", "DE", "STCPNQ")
AA_TO_GROUP = {aa: i for i, group in enumerate(AA_GROUPS) for aa in group}

PSEAAC_PROPERTIES = np.asarray(
    [
        [0.62, -0.5, 15.0], [0.29, -1.0, 47.0], [-0.90, 3.0, 59.0],
        [-0.74, 3.0, 73.0], [1.19, -2.5, 91.0], [0.48, 0.0, 1.0],
        [-0.40, -0.5, 82.0], [1.38, -1.8, 57.0], [-1.50, 3.0, 73.0],
        [1.06, -1.8, 57.0], [0.64, -1.3, 75.0], [-0.78, 0.2, 58.0],
        [0.12, 0.0, 42.0], [-0.85, 0.2, 72.0], [-2.53, 3.0, 101.0],
        [-0.18, 0.3, 31.0], [-0.05, -0.4, 45.0], [1.08, -1.5, 43.0],
        [0.81, -3.4, 130.0], [0.26, -2.3, 107.0],
    ],
    dtype=np.float64,
)
PSEAAC_PROPERTIES = (
    PSEAAC_PROPERTIES - PSEAAC_PROPERTIES.mean(axis=0, keepdims=True)
) / PSEAAC_PROPERTIES.std(axis=0, ddof=0, keepdims=True)


def amino_acid_composition(sequence: str) -> np.ndarray:
    indices = np.fromiter((AA_TO_INDEX[aa] for aa in sequence), dtype=np.int64)
    return np.bincount(indices, minlength=20).astype(np.float64) / len(sequence)


def dipeptide_composition(sequence: str) -> np.ndarray:
    values = np.zeros((20, 20), dtype=np.float64)
    for left, right in zip(sequence[:-1], sequence[1:]):
        values[AA_TO_INDEX[left], AA_TO_INDEX[right]] += 1.0
    return (values / max(len(sequence) - 1, 1)).reshape(-1)


def pseudo_amino_acid_composition(
    sequence: str, lambda_value: int = 5, weight: float = 0.05
) -> np.ndarray:
    indices = np.fromiter((AA_TO_INDEX[aa] for aa in sequence), dtype=np.int64)
    frequencies = np.bincount(indices, minlength=20).astype(np.float64) / len(sequence)
    theta = np.zeros(lambda_value, dtype=np.float64)
    for lag in range(1, lambda_value + 1):
        if lag < len(sequence):
            delta = PSEAAC_PROPERTIES[indices[:-lag]] - PSEAAC_PROPERTIES[indices[lag:]]
            theta[lag - 1] = np.square(delta).mean()
    denominator = 1.0 + weight * theta.sum()
    return np.concatenate((frequencies / denominator, weight * theta / denominator))


def cksaagp(sequence: str, max_gap: int = 2) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for gap in range(max_gap + 1):
        offset = gap + 1
        counts = np.zeros((5, 5), dtype=np.float64)
        for left, right in zip(sequence[:-offset], sequence[offset:]):
            counts[AA_TO_GROUP[left], AA_TO_GROUP[right]] += 1.0
        blocks.append((counts / max(len(sequence) - offset, 1)).reshape(-1))
    return np.concatenate(blocks)


def extract_one(sequence: str) -> np.ndarray:
    return np.concatenate(
        [
            amino_acid_composition(sequence),
            dipeptide_composition(sequence),
            pseudo_amino_acid_composition(sequence),
            cksaagp(sequence),
        ]
    ).astype(np.float32)


def extract_features(sequences: Sequence[str]) -> np.ndarray:
    result = np.stack([extract_one(sequence) for sequence in sequences])
    if result.shape[1] != 520:
        raise AssertionError(f"Expected 520 features; got {result.shape[1]}")
    return result


def cksaap(sequence: str, max_gap: int = 8) -> np.ndarray:
    """Composition of k-spaced amino-acid pairs for gaps 0..``max_gap``.

    Every 400-value gap block is normalized by the number of residue pairs
    available at that gap.  A zero block is returned when a peptide is too
    short, which keeps the representation defined for every benchmark member.
    The IEEE Access ACP-KSRC study used gaps 0--8 (3,600 values).
    """
    blocks: list[np.ndarray] = []
    for gap in range(max_gap + 1):
        offset = gap + 1
        counts = np.zeros((20, 20), dtype=np.float64)
        pair_count = max(len(sequence) - offset, 0)
        if pair_count:
            for left, right in zip(sequence[:-offset], sequence[offset:]):
                counts[AA_TO_INDEX[left], AA_TO_INDEX[right]] += 1.0
            counts /= pair_count
        blocks.append(counts.reshape(-1))
    return np.concatenate(blocks).astype(np.float32)


def extract_cksaap(sequences: Sequence[str], max_gap: int = 8) -> np.ndarray:
    """Return CKSAAP features with ``400 * (max_gap + 1)`` columns."""
    result = np.stack([cksaap(sequence, max_gap=max_gap) for sequence in sequences])
    expected = 400 * (max_gap + 1)
    if result.shape[1] != expected:
        raise AssertionError(f"Expected {expected} CKSAAP features; got {result.shape[1]}")
    return result


def split_antidmppred_views(features: np.ndarray) -> dict[str, np.ndarray]:
    """Split the audited 520-vector into its four biological descriptor views."""
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 520:
        raise ValueError("AntiDMPpred feature matrix must have shape (n, 520)")
    return {
        "AAC": features[:, :20],
        "DPC": features[:, 20:420],
        "PseAAC": features[:, 420:445],
        "CKSAAGP": features[:, 445:520],
    }


def pearson_ranking(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_x = x - x.mean(axis=0, keepdims=True)
    centered_y = y.astype(np.float64) - y.mean()
    numerator = centered_x.T @ centered_y
    denominator = np.sqrt((centered_x**2).sum(axis=0) * (centered_y**2).sum())
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return np.argsort(-np.abs(np.nan_to_num(scores)), kind="stable")
