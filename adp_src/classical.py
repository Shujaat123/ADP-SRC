from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA, sparse_encode
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC

from .features import pearson_ranking


@dataclass
class SRCTransformer:
    mode: str = "pearson347"
    seed: int = 6

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SRCTransformer":
        x = np.asarray(x, dtype=np.float64)
        if self.mode == "aac":
            self.indices_ = np.arange(20)
        elif self.mode.startswith("pearson"):
            count = int(self.mode.replace("pearson", ""))
            self.indices_ = pearson_ranking(x, y)[:count]
        elif self.mode.startswith("pca"):
            self.indices_ = np.arange(x.shape[1])
        else:
            raise ValueError(f"Unknown SRC feature mode: {self.mode}")
        selected = x[:, self.indices_]
        self.scaler_ = StandardScaler().fit(selected)
        selected = self.scaler_.transform(selected)
        if self.mode.startswith("pca"):
            components = int(self.mode.replace("pca", ""))
            self.pca_ = PCA(
                n_components=min(components, len(x) - 1, selected.shape[1]),
                random_state=self.seed,
            ).fit(selected)
        else:
            self.pca_ = None
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        selected = self.scaler_.transform(np.asarray(x)[:, self.indices_])
        if self.pca_ is not None:
            selected = self.pca_.transform(selected)
        return normalize(selected, norm="l2", axis=1)


def src_predict_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify by class-specific reconstruction residuals.

    ``sparse_encode`` solves the standard L1 sparse-coding objective. Rows of
    ``x_train`` are dictionary atoms; the final score is the log residual ratio,
    so 0.5 corresponds exactly to the lower-residual decision rule.
    """
    # Highly correlated peptide atoms can make the LARS path non-unique. The
    # returned sparse optimum remains usable; suppress only that known warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        code = sparse_encode(
            x_test,
            x_train,
            algorithm="lasso_lars",
            alpha=float(alpha),
            max_iter=1000,
            n_jobs=-1,
        )
    residuals = np.zeros((len(x_test), 2), dtype=np.float64)
    for label in (0, 1):
        keep = y_train == label
        reconstruction = code[:, keep] @ x_train[keep]
        residuals[:, label] = np.linalg.norm(x_test - reconstruction, axis=1)
    score = np.log((residuals[:, 0] + 1e-12) / (residuals[:, 1] + 1e-12))
    score = np.clip(score, -40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-score))
    return probability, residuals, code


def tune_src(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    modes: tuple[str, ...] = ("aac", "pca64", "pearson128", "pearson347"),
    alphas: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02),
    inner_splits: int = 3,
    groups: np.ndarray | None = None,
) -> tuple[str, float, list[dict[str, float | str]]]:
    if groups is None:
        splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        split_iterator = lambda: splitter.split(x, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        split_iterator = lambda: splitter.split(x, y, groups=groups)
    scored: list[dict[str, float | str]] = []
    for mode in modes:
        for alpha in alphas:
            truth: list[int] = []
            prediction: list[int] = []
            for fold, (train, valid) in enumerate(split_iterator()):
                transformer = SRCTransformer(mode=mode, seed=seed + fold).fit(x[train], y[train])
                train_x = transformer.transform(x[train])
                valid_x = transformer.transform(x[valid])
                probability, _, _ = src_predict_proba(train_x, y[train], valid_x, alpha)
                truth.extend(y[valid])
                prediction.extend(probability >= 0.5)
            mcc = float(matthews_corrcoef(truth, prediction))
            scored.append({"mode": mode, "alpha": alpha, "mcc": mcc})
    best = max(scored, key=lambda row: (float(row["mcc"]), -modes.index(str(row["mode"])), -alphas.index(float(row["alpha"]))))
    return str(best["mode"]), float(best["alpha"]), scored


def fit_reference_rf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = pearson_ranking(x_train, y_train)[:347]
    model = RandomForestClassifier(
        n_estimators=251,
        criterion="gini",
        bootstrap=True,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train[:, indices], y_train)
    return model.predict_proba(x_test[:, indices])[:, 1], indices


def fit_rbf_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(x_train)
    model = SVC(C=3.0, gamma="scale", probability=True, random_state=seed)
    model.fit(scaler.transform(x_train), y_train)
    return model.predict_proba(scaler.transform(x_test))[:, 1]


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Standardized L2-logistic regression baseline."""
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )
    model.fit(scaler.transform(x_train), y_train)
    return model.predict_proba(scaler.transform(x_test))[:, 1]


def fit_gradient_boosting(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Conventional gradient-boosted decision-tree baseline on all descriptors."""
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=1.0,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]
