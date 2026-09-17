import numpy as np
import pytest

pytest.importorskip("torch", reason="DeepLSE tests require the optional neural dependency")
from adp_src.deep_lse import DeepLSEConfig, encode_lse, select_and_refit_lse


def test_deeplse_small_refit() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(size=(40, 16)).astype(np.float32)
    y = np.asarray([0, 1] * 20)
    x[y == 1, :3] += 0.8
    config = DeepLSEConfig(
        input_dim=16,
        hidden_dim_1=12,
        hidden_dim_2=8,
        latent_dim=4,
        dropout=0.0,
        batch_size=16,
        epochs=2,
        patience=1,
    )
    model, scaler, metadata, rows = select_and_refit_lse(
        x,
        y,
        seed=10,
        device="cpu",
        base_config=config,
        latent_dims=(4,),
        alphas=(0.01,),
    )
    z = encode_lse(model, scaler, x, "cpu")
    assert z.shape == (40, 4)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)
    assert metadata["refit_on_complete_outer_training_fold"] is True
    assert len(rows) == 1
