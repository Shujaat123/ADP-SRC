"""Kernel embeddings and sparse-reconstruction classifiers for peptides."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import KernelPCA, sparse_encode
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.preprocessing import StandardScaler, normalize


def _center_square_kernel(kernel: np.ndarray) -> np.ndarray:
    row_mean = kernel.mean(axis=1, keepdims=True)
    col_mean = kernel.mean(axis=0, keepdims=True)
    return kernel - row_mean - col_mean + kernel.mean()


def centered_alignment(kernel: np.ndarray, y: np.ndarray) -> float:
    """Centered Frobenius alignment with the binary target kernel."""
    labels = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
    target = np.outer(labels, labels)
    centered_kernel = _center_square_kernel(np.asarray(kernel, dtype=np.float64))
    centered_target = _center_square_kernel(target)
    denominator = np.linalg.norm(centered_kernel) * np.linalg.norm(centered_target)
    return float(np.sum(centered_kernel * centered_target) / denominator) if denominator else 0.0


def _median_gamma(x: np.ndarray) -> float:
    distances = pdist(np.asarray(x, dtype=np.float64), metric="sqeuclidean")
    positive = distances[distances > 0]
    median = float(np.median(positive)) if len(positive) else 1.0
    return 1.0 / max(median, 1e-12)


@dataclass
class PolynomialCKSAAPEmbedding:
    """Train-only standardization followed by polynomial-kernel PCA."""

    n_components: int = 64
    degree: int = 2
    seed: int = 6

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "PolynomialCKSAAPEmbedding":
        del y
        self.scaler_ = StandardScaler().fit(x)
        scaled = self.scaler_.transform(x)
        # The explicit precomputed kernel makes all numerical settings auditable.
        self.gamma_ = 1.0 / scaled.shape[1]
        kernel = polynomial_kernel(
            scaled, scaled, degree=self.degree, gamma=self.gamma_, coef0=1.0
        )
        count = min(self.n_components, len(x) - 1)
        self.kpca_ = KernelPCA(
            n_components=count,
            kernel="precomputed",
            eigen_solver="arpack",
            remove_zero_eig=True,
            random_state=self.seed,
        ).fit(kernel)
        self.train_scaled_ = scaled
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        scaled = self.scaler_.transform(x)
        cross_kernel = polynomial_kernel(
            scaled,
            self.train_scaled_,
            degree=self.degree,
            gamma=self.gamma_,
            coef0=1.0,
        )
        return normalize(self.kpca_.transform(cross_kernel), norm="l2", axis=1)


@dataclass
class AlignmentWeightedMultiKernelEmbedding:
    """Alignment-weighted RBF-kernel fusion followed by kernel PCA.

    View-specific scalers, RBF bandwidths and alignment weights are estimated
    only from the supplied training fold.  This makes the supervised embedding
    safe to use inside nested cross-validation.
    """

    n_components: int = 64
    seed: int = 6

    def fit(
        self, views: dict[str, np.ndarray], y: np.ndarray
    ) -> "AlignmentWeightedMultiKernelEmbedding":
        self.view_names_ = tuple(sorted(views))
        self.scalers_: dict[str, StandardScaler] = {}
        self.gammas_: dict[str, float] = {}
        self.train_scaled_: dict[str, np.ndarray] = {}
        kernels: list[np.ndarray] = []
        alignments: list[float] = []
        for name in self.view_names_:
            scaler = StandardScaler().fit(views[name])
            scaled = scaler.transform(views[name])
            gamma = _median_gamma(scaled)
            kernel = rbf_kernel(scaled, scaled, gamma=gamma)
            self.scalers_[name] = scaler
            self.gammas_[name] = gamma
            self.train_scaled_[name] = scaled
            kernels.append(kernel)
            alignments.append(max(centered_alignment(kernel, y), 0.0))
        weights = np.asarray(alignments, dtype=np.float64)
        if not np.any(weights):
            weights = np.ones_like(weights)
        self.weights_ = weights / weights.sum()
        fused = sum(weight * kernel for weight, kernel in zip(self.weights_, kernels))
        count = min(self.n_components, len(y) - 1)
        self.kpca_ = KernelPCA(
            n_components=count,
            kernel="precomputed",
            eigen_solver="arpack",
            remove_zero_eig=True,
            random_state=self.seed,
        ).fit(fused)
        return self

    def transform(self, views: dict[str, np.ndarray]) -> np.ndarray:
        fused = None
        for weight, name in zip(self.weights_, self.view_names_):
            scaled = self.scalers_[name].transform(views[name])
            kernel = rbf_kernel(
                scaled, self.train_scaled_[name], gamma=self.gammas_[name]
            )
            fused = weight * kernel if fused is None else fused + weight * kernel
        return normalize(self.kpca_.transform(fused), norm="l2", axis=1)

    @property
    def weight_map(self) -> dict[str, float]:
        return dict(zip(self.view_names_, map(float, self.weights_)))


def matching_pursuit_code(
    dictionary: np.ndarray,
    samples: np.ndarray,
    max_atoms: int,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Greedy matching pursuit with row-wise unit-norm dictionary atoms."""
    dictionary = normalize(np.asarray(dictionary, dtype=np.float64), axis=1)
    samples = np.asarray(samples, dtype=np.float64)
    codes = np.zeros((len(samples), len(dictionary)), dtype=np.float64)
    for sample_index, sample in enumerate(samples):
        residual = sample.copy()
        initial_norm = max(np.linalg.norm(residual), 1e-12)
        for _ in range(int(max_atoms)):
            correlations = dictionary @ residual
            atom = int(np.argmax(np.abs(correlations)))
            step = correlations[atom]
            if abs(step) < 1e-12:
                break
            codes[sample_index, atom] += step
            residual -= step * dictionary[atom]
            if np.linalg.norm(residual) <= tolerance * initial_norm:
                break
    return codes


def _residual_probability(
    train: np.ndarray, y_train: np.ndarray, test: np.ndarray, code: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.zeros((len(test), 2), dtype=np.float64)
    for label in (0, 1):
        keep = np.asarray(y_train) == label
        reconstruction = code[:, keep] @ train[keep]
        residuals[:, label] = np.linalg.norm(test - reconstruction, axis=1)
    score = np.log((residuals[:, 0] + 1e-12) / (residuals[:, 1] + 1e-12))
    probability = 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))
    return probability, residuals


def mp_src_predict_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    max_atoms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    code = matching_pursuit_code(x_train, x_test, max_atoms=max_atoms)
    probability, residuals = _residual_probability(x_train, y_train, x_test, code)
    return probability, residuals, code


def l1_src_predict_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    probability, residuals = _residual_probability(x_train, y_train, x_test, code)
    return probability, residuals, code
