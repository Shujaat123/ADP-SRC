from pathlib import Path

import numpy as np

from adp_src.classical import SRCTransformer, src_predict_proba
from adp_src.data import read_labeled_fasta, records_to_arrays
from adp_src.features import extract_features
from adp_src.splits import make_outer_splits


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_and_features() -> None:
    records = read_labeled_fasta(ROOT / "data/positive_negative.fasta")
    sequences, y, identifiers = records_to_arrays(records)
    x = extract_features(sequences)
    assert x.shape == (472, 520)
    assert len(identifiers) == 472
    assert np.bincount(y).tolist() == [236, 236]
    assert np.isfinite(x).all()


def test_pca_src_dictionary_is_overcomplete() -> None:
    records = read_labeled_fasta(ROOT / "data/positive_negative.fasta")
    sequences, y, _ = records_to_arrays(records)
    x = extract_features(sequences[:100])
    transformer = SRCTransformer(mode="pca64", seed=6).fit(x[:80], y[:80])
    dictionary = transformer.transform(x[:80])
    probability, residuals, code = src_predict_proba(
        dictionary, y[:80], transformer.transform(x[80:]), 0.01
    )
    assert dictionary.shape == (80, 64)
    assert dictionary.shape[0] > dictionary.shape[1]
    assert probability.shape == (20,)
    assert residuals.shape == (20, 2)
    assert code.shape == (20, 80)


def test_direct_and_grouped_outer_splits() -> None:
    records = read_labeled_fasta(ROOT / "data/positive_negative.fasta")
    _, y, identifiers = records_to_arrays(records)
    direct, _ = make_outer_splits(y, identifiers, "direct", seed=6)
    grouped, groups = make_outer_splits(
        y,
        identifiers,
        "cluster",
        seed=6,
        cluster_file=ROOT / "data/positive_negative_cdhit50.fasta.clstr",
    )
    assert len(direct) == len(grouped) == 5
    for train, test in grouped:
        assert set(groups[train]).isdisjoint(groups[test])
