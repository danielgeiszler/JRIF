"""Save → load → predict round-trip preserves predictions exactly."""

import numpy as np
import polars as pl

from jrif.api import RTPredictor


def _toy_df(n=64, seed=0):
    rng = np.random.default_rng(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    seqs = []
    for _ in range(n):
        L = int(rng.integers(6, 16))
        seqs.append("".join(rng.choice(aas, size=L)))
    rts = rng.normal(50.0, 20.0, size=n).astype(np.float32)
    return pl.DataFrame({"seq": seqs, "rt": rts})


def test_save_load_predictions_match(tmp_path):
    df = _toy_df()
    predictor = RTPredictor.from_blank()
    predictor.fit(
        df, epochs=1, batch_size=16, warmup_epochs=0, lr=1e-3,
        seed=0, val_fraction=0.2, verbose=False,
    )
    pred_before = predictor.predict(df)
    p = tmp_path / "ckpt.pt"
    predictor.save(p)

    loaded = RTPredictor.from_checkpoint(p)
    pred_after = loaded.predict(df)
    np.testing.assert_allclose(pred_before, pred_after, rtol=0, atol=1e-5)
    assert loaded.rt_median == predictor.rt_median
    assert loaded.rt_mad == predictor.rt_mad
    assert loaded.vocab.tokens == predictor.vocab.tokens


def test_save_load_with_extended_vocab(tmp_path):
    """Fit on data containing a novel mod → vocab grows → save → load round-trip
    preserves the extended vocab and predictions."""
    rng = np.random.default_rng(1)
    df = pl.DataFrame({
        "seq": ["AAS[80]K", "PEPS[80]TIDE", "AAAK", "MCAR"] * 8,
        "rt": rng.normal(50, 5, size=32).astype(np.float32),
    })
    predictor = RTPredictor.from_blank()
    predictor.fit(df, epochs=1, batch_size=8, warmup_epochs=0, lr=1e-3,
                  seed=0, val_fraction=0.25, verbose=False)
    assert "S[+80]" in predictor.vocab.tokens

    p = tmp_path / "ckpt.pt"
    predictor.save(p)

    loaded = RTPredictor.from_checkpoint(p)
    assert "S[+80]" in loaded.vocab.tokens
    np.testing.assert_allclose(predictor.predict(df), loaded.predict(df), rtol=0, atol=1e-5)
