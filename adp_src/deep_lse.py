from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .classical import src_predict_proba


@dataclass(frozen=True)
class DeepLSEConfig:
    """Leakage-safe DeepLSE configuration for the 520 peptide descriptors."""

    input_dim: int = 520
    hidden_dim_1: int = 256
    hidden_dim_2: int = 128
    latent_dim: int = 64
    dropout: float = 0.30
    classification_weight: float = 1.0
    learning_rate: float = 8.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 64
    epochs: int = 100
    patience: int = 14
    validation_fraction: float = 0.20


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class DeepLSE(nn.Module):
    """Encoder-decoder-classifier latent-space engineering network."""

    def __init__(self, config: DeepLSEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim_1),
            nn.LayerNorm(config.hidden_dim_1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim_1, config.hidden_dim_2),
            nn.LayerNorm(config.hidden_dim_2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim_2, config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim_2),
            nn.GELU(),
            nn.Linear(config.hidden_dim_2, config.hidden_dim_1),
            nn.GELU(),
            nn.Linear(config.hidden_dim_1, config.input_dim),
        )
        self.classifier = nn.Linear(config.latent_dim, 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(features)
        reconstruction = self.decoder(latent)
        logits = self.classifier(latent)
        return latent, reconstruction, logits


def _split_selection(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    validation_fraction: float,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if groups is None:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=validation_fraction, random_state=seed
        )
        return next(splitter.split(x, y))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return next(splitter.split(x, y, groups=groups))


def _loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _loss(
    model: DeepLSE, features: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, reconstruction, logits = model(features)
    reconstruction_loss = F.mse_loss(reconstruction, features)
    classification_loss = F.cross_entropy(logits, labels)
    total = reconstruction_loss + model.config.classification_weight * classification_loss
    return total, reconstruction_loss, classification_loss


@torch.no_grad()
def encode_lse(
    model: DeepLSE,
    scaler: StandardScaler,
    features: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    scaled = scaler.transform(np.asarray(features)).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(scaled, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    blocks: list[np.ndarray] = []
    model.eval()
    for (batch,) in loader:
        latent, _, _ = model(batch.to(device))
        blocks.append(latent.cpu().numpy())
    return normalize(np.concatenate(blocks), norm="l2", axis=1)


@torch.no_grad()
def predict_lse_classifier(
    model: DeepLSE,
    scaler: StandardScaler,
    features: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    scaled = scaler.transform(np.asarray(features)).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(scaled, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    blocks: list[np.ndarray] = []
    model.eval()
    for (batch,) in loader:
        logits = model(batch.to(device))[2]
        blocks.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(blocks)


def _fit_fixed_epochs(
    model: DeepLSE,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
) -> None:
    loader = _loader(x, y, model.config.batch_size, seed, True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model.config.learning_rate,
        weight_decay=model.config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            total, _, _ = _loss(model, batch_x.to(device), batch_y.to(device))
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()


def _fit_selection_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    config: DeepLSEConfig,
    seed: int,
    device: str,
) -> tuple[DeepLSE, list[dict[str, float]], int]:
    seed_everything(seed)
    model = DeepLSE(config).to(device)
    loader = _loader(x_train, y_train, config.batch_size, seed, True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    valid_x = torch.as_tensor(x_valid, dtype=torch.float32, device=device)
    valid_y = torch.as_tensor(y_valid, dtype=torch.long, device=device)
    best_value = float("inf")
    best_mcc = -float("inf")
    best_auc = -float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    patience = config.patience
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            total, _, _ = _loss(model, batch_x.to(device), batch_y.to(device))
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(total.detach()) * len(batch_y)
            seen += len(batch_y)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            total, recon, cls = _loss(model, valid_x, valid_y)
            logits = model(valid_x)[2]
            probability = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        mcc = float(matthews_corrcoef(y_valid, probability >= 0.5))
        auc = float(roc_auc_score(y_valid, probability))
        value = float(recon)
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": running / max(seen, 1),
                "validation_loss": float(total),
                "validation_reconstruction_loss": float(recon),
                "validation_classification_loss": float(cls),
                "validation_classifier_mcc": mcc,
                "validation_classifier_auc": auc,
            }
        )
        if config.classification_weight > 0:
            improved = (mcc > best_mcc + 1.0e-8) or (
                abs(mcc - best_mcc) <= 1.0e-8 and auc > best_auc + 1.0e-8
            )
        else:
            improved = value < best_value - 1.0e-6
        if improved:
            best_value = value
            best_mcc = mcc
            best_auc = auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = config.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch


def select_and_refit_lse(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    device: str,
    base_config: DeepLSEConfig,
    latent_dims: tuple[int, ...] = (32, 64, 128),
    alphas: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02),
    groups: np.ndarray | None = None,
) -> tuple[DeepLSE, StandardScaler, dict[str, object], list[dict[str, object]]]:
    """Nested selection of latent width and SRC alpha, then full-fold refit."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    selection_train, selection_valid = _split_selection(
        features, labels, seed, base_config.validation_fraction, groups
    )
    scaler = StandardScaler().fit(features[selection_train])
    train_x = scaler.transform(features[selection_train]).astype(np.float32)
    valid_x = scaler.transform(features[selection_valid]).astype(np.float32)
    rows: list[dict[str, object]] = []
    candidates: list[tuple[tuple[float, float, float], int, float, int]] = []

    for latent_dim in latent_dims:
        config = replace(base_config, latent_dim=latent_dim)
        model, history, selected_epochs = _fit_selection_model(
            train_x,
            labels[selection_train],
            valid_x,
            labels[selection_valid],
            config,
            seed + latent_dim,
            device,
        )
        train_z = encode_lse(model, scaler, features[selection_train], device)
        valid_z = encode_lse(model, scaler, features[selection_valid], device)
        for alpha in alphas:
            probability, _, _ = src_predict_proba(
                train_z, labels[selection_train], valid_z, alpha
            )
            mcc = float(matthews_corrcoef(labels[selection_valid], probability >= 0.5))
            auc = float(roc_auc_score(labels[selection_valid], probability))
            rows.append(
                {
                    "latent_dim": latent_dim,
                    "alpha": alpha,
                    "selected_epochs": selected_epochs,
                    "inner_mcc": mcc,
                    "inner_auc": auc,
                    "classification_weight": config.classification_weight,
                    "history_last_epoch": len(history),
                }
            )
            candidates.append(((mcc, auc, -float(latent_dim)), latent_dim, alpha, selected_epochs))

    _, latent_dim, alpha, selected_epochs = max(candidates, key=lambda item: item[0])
    selected_config = replace(base_config, latent_dim=latent_dim)
    full_scaler = StandardScaler().fit(features)
    full_x = full_scaler.transform(features).astype(np.float32)
    seed_everything(seed + latent_dim)
    final_model = DeepLSE(selected_config).to(device)
    _fit_fixed_epochs(
        final_model, full_x, labels, seed + latent_dim, device, selected_epochs
    )
    metadata: dict[str, object] = {
        **asdict(selected_config),
        "selected_alpha": alpha,
        "selected_epochs": selected_epochs,
        "selection_train_size": len(selection_train),
        "selection_valid_size": len(selection_valid),
        "selection_group_aware": groups is not None,
        "epoch_selection_criterion": (
            "validation classifier MCC, then AUC"
            if selected_config.classification_weight > 0
            else "validation reconstruction MSE"
        ),
        "refit_on_complete_outer_training_fold": True,
        "seed": seed,
    }
    return final_model, full_scaler, metadata, rows


def fit_latent_heads(
    train_z: np.ndarray,
    y_train: np.ndarray,
    test_z: np.ndarray,
    alpha: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    src_probability, residuals, coefficients = src_predict_proba(
        train_z, y_train, test_z, alpha
    )
    svm = SVC(C=3.0, gamma="scale", probability=True, random_state=seed)
    svm.fit(train_z, y_train)
    rf = RandomForestClassifier(
        n_estimators=251,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(train_z, y_train)
    return (
        {
            "DeepLSE-SRC": src_probability,
            "DeepLSE-SVM": svm.predict_proba(test_z)[:, 1],
            "DeepLSE-RF": rf.predict_proba(test_z)[:, 1],
        },
        residuals,
        coefficients,
    )
