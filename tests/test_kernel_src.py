from pathlib import Path

import joblib
import numpy as np

from adp_src.data import read_labeled_fasta, records_to_arrays
from adp_src.features import extract_cksaap, extract_features, split_antidmppred_views
from adp_src.kernel_src import (
    AlignmentWeightedMultiKernelEmbedding,
    PolynomialCKSAAPEmbedding,
    l1_src_predict_proba,
    matching_pursuit_code,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cksaap_shape_and_short_sequence_blocks() -> None:
    x = extract_cksaap(["ACDEF", "ACDEFGHIKLM"], max_gap=8)
    assert x.shape == (2, 3600)
    assert np.isfinite(x).all()
    assert np.allclose(x[0, 4 * 400 :], 0.0)


def test_polynomial_dictionary_and_sparse_codes() -> None:
    records = read_labeled_fasta(ROOT / "data/positive_negative.fasta")[:90]
    sequences, y, _ = records_to_arrays(records)
    x = extract_cksaap(sequences)
    embedding = PolynomialCKSAAPEmbedding(n_components=16, seed=3).fit(x[:70], y[:70])
    dictionary = embedding.transform(x[:70])
    test = embedding.transform(x[70:])
    assert dictionary.shape == (70, 16)
    assert dictionary.shape[0] > dictionary.shape[1]
    mp_code = matching_pursuit_code(dictionary, test, max_atoms=5)
    assert mp_code.shape == (20, 70)
    probability, residuals, l1_code = l1_src_predict_proba(
        dictionary, y[:70], test, alpha=0.005
    )
    assert probability.shape == (20,)
    assert residuals.shape == (20, 2)
    assert l1_code.shape == (20, 70)


def test_alignment_weights_are_training_derived_probabilities() -> None:
    records = read_labeled_fasta(ROOT / "data/positive_negative.fasta")[:120]
    sequences, y, _ = records_to_arrays(records)
    views = split_antidmppred_views(extract_features(sequences))
    embedding = AlignmentWeightedMultiKernelEmbedding(n_components=16, seed=4).fit(views, y)
    weights = embedding.weight_map
    assert set(weights) == {"AAC", "CKSAAGP", "DPC", "PseAAC"}
    assert np.isclose(sum(weights.values()), 1.0)
    assert all(value >= 0 for value in weights.values())
    assert embedding.transform(views).shape == (120, 16)


def test_final_kernel_model_is_overcomplete() -> None:
    model = joblib.load(ROOT / "final_model/kernel_src/cksaap_kpca_l1_src.joblib")
    assert model["model_name"] == "CKSAAP-KPCA-L1-SRC"
    assert model["dictionary"].shape == (472, 64)
    assert len(model["dictionary_labels"]) == 472
    assert model["dictionary"].shape[0] > model["dictionary"].shape[1]
