from pathlib import Path

import numpy as np
import pandas as pd

from adp_src.classical import SRCTransformer, fit_gradient_boosting, fit_logistic_regression


ROOT = Path(__file__).resolve().parents[1]


def test_generic_pca160_dictionary_is_overcomplete() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(180, 520))
    y = np.asarray([0, 1] * 90)
    transformer = SRCTransformer(mode="pca160", seed=13).fit(x, y)
    dictionary = transformer.transform(x)
    assert dictionary.shape == (180, 160)
    assert dictionary.shape[0] > dictionary.shape[1]
    assert np.allclose(np.linalg.norm(dictionary, axis=1), 1.0)


def test_new_conventional_comparators_return_probabilities() -> None:
    rng = np.random.default_rng(14)
    train_x = rng.normal(size=(80, 20))
    train_y = np.asarray([0, 1] * 40)
    test_x = rng.normal(size=(12, 20))
    for predictor in (fit_logistic_regression, fit_gradient_boosting):
        probability = predictor(train_x, train_y, test_x, seed=14)
        assert probability.shape == (12,)
        assert np.isfinite(probability).all()
        assert ((0 <= probability) & (probability <= 1)).all()


def test_saved_extended_pca_ablation() -> None:
    frame = pd.read_csv(
        ROOT / "results/revised_pca_dimension_ablation_extended_5x5/aggregate_metrics.csv"
    )
    assert frame["dimension"].tolist() == [16, 32, 64, 96, 128, 160, 192, 224, 240]
    assert int(frame.loc[frame["f1_mean"].idxmax(), "dimension"]) == 240
    pca160 = frame[frame["dimension"] == 160].iloc[0]
    pca240 = frame[frame["dimension"] == 240].iloc[0]
    assert pca240["f1_mean"] - pca160["f1_mean"] < 0.005
    assert pca160["mcc_mean"] == frame["mcc_mean"].max()
    assert pca160["auc_roc_mean"] == frame["auc_roc_mean"].max()
    assert pca160["f1_std"] < pca240["f1_std"]


def test_saved_common_width_src_comparison() -> None:
    frame = pd.read_csv(
        ROOT / "results/revised_src_variants_160_5x5/aggregate_metrics.csv"
    )
    assert set(frame["method"]) == {
        "PCA160-SRC (proposed)",
        "DeepLSE160-SRC",
        "AE160-SRC",
        "CKSAAP-KPCA160-MP",
        "CKSAAP-KPCA160-L1-SRC",
        "AWMKPCA160-L1-SRC",
    }
    pca = frame[frame["method"] == "PCA160-SRC (proposed)"].iloc[0]
    kernel = frame[frame["method"] == "CKSAAP-KPCA160-L1-SRC"].iloc[0]
    awmk = frame[frame["method"] == "AWMKPCA160-L1-SRC"].iloc[0]
    assert pca["accuracy_mean"] == frame["accuracy_mean"].max()
    assert pca["mcc_mean"] == frame["mcc_mean"].max()
    assert abs(pca["auc_roc_mean"] - kernel["auc_roc_mean"]) < 0.001
    assert awmk["f1_mean"] > pca["f1_mean"]
