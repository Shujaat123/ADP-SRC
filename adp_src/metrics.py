from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, probability)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def bootstrap_interval(
    y_true: np.ndarray,
    probability: np.ndarray,
    metric: str,
    seed: int = 2026,
    n_bootstrap: int = 2000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        values.append(float(classification_metrics(y_true[indices], probability[indices])[metric]))
    return tuple(np.quantile(values, [0.025, 0.975]))
