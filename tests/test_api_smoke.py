"""End-to-end smoke tests for RTPredictor on tiny synthetic DataFrames."""

import re
import numpy as np
import pandas as pd
import polars as pl

from jrif.api import RTPredictor


def _toy(n=64, seed=0):
    rng = np.random.default_rng(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    seqs = []
    for _ in range(n):
        L = int(rng.integers(6, 16))
        seqs.append("".join(rng.choice(aas, size=L)))
    rts = (np.array([len(s) for s in seqs], dtype=np.float32) * 5.0
           + rng.normal(0, 2, size=n).astype(np.float32))
    return seqs, rts


def test_fit_predict_polars_input():
    seqs, rts = _toy()
    df = pl.DataFrame({"seq": seqs, "rt": rts})
    predictor = RTPredictor.from_blank()
    result = predictor.fit(df, epochs=2, batch_size=16, warmup_epochs=0,
                            lr=1e-3, seed=0, verbose=False)
    assert np.isfinite(result.best_val_r2)
    preds = predictor.predict(df)
    assert preds.shape == (len(df),)
    assert np.all(np.isfinite(preds))


def test_fit_predict_pandas_input():
    seqs, rts = _toy()
    df = pd.DataFrame({"seq": seqs, "rt": rts})
    predictor = RTPredictor.from_blank()
    predictor.fit(df, epochs=2, batch_size=16, warmup_epochs=0,
                  lr=1e-3, seed=0, verbose=False)
    preds = predictor.predict(df)
    assert preds.shape == (len(df),)
    assert np.all(np.isfinite(preds))


def test_fit_extends_vocab_for_new_mod():
    rng = np.random.default_rng(2)
    suffixes = ["A", "R", "P", "L", "F"]
    seqs = [f"AS[80]K{suffixes[i % len(suffixes)]}EPTIDE" for i in range(48)]
    rts = rng.normal(50, 5, size=len(seqs)).astype(np.float32)
    df = pl.DataFrame({"seq": seqs, "rt": rts})

    predictor = RTPredictor.from_blank()
    assert "S[+80]" not in predictor.vocab.tokens
    predictor.fit(df, epochs=1, batch_size=16, warmup_epochs=0, lr=1e-3,
                  seed=0, val_fraction=0.2, verbose=False)
    assert "S[+80]" in predictor.vocab.tokens
    assert predictor.vocab.parent_of.get("S[+80]") == "S"

    preds = predictor.predict(df)
    assert preds.shape == (len(df),)


def test_fit_detects_and_logs_every_mod(capsys):
    """No declaration needed: a new residue mod starts from its residue, a
    new terminal mod from <nterm>, and the log lists every mod exactly as
    written, with the token it resolved to when that differs."""
    rng = np.random.default_rng(3)
    seqs = ["AS[80]K", ".[42]PEPS[80]TIDE", "AAAK", "M[147]CAR"] * 8
    rts = rng.normal(50, 5, size=len(seqs)).astype(np.float32)
    df = pl.DataFrame({"seq": seqs, "rt": rts})
    predictor = RTPredictor.from_blank()
    predictor.fit(df, epochs=1, batch_size=8, warmup_epochs=0, lr=1e-3,
                  seed=0, val_fraction=0.25, verbose=True)
    v = predictor.vocab
    assert v.parent_of["S[+80]"] == "S" and "S[+80]" not in v.fresh
    assert v.parent_of["n[+42]"] == "<nterm>" and "n[+42]" not in v.fresh
    log = capsys.readouterr().out
    assert "3 modification(s) in data, 2 new" in log
    assert re.search(r"M\[147\]\s+8 IDs  known \(as M\[\+15\.9949\]\)", log)
    assert re.search(r"S\[80\]\s+16 IDs  NEW, initialized from S \(as S\[\+80\]\)", log)
    assert re.search(r"\.\[42\]\s+8 IDs  NEW, initialized from <nterm> \(as n\[\+42\]\)", log)

def test_explicit_val_df():
    seqs, rts = _toy(n=80, seed=4)
    train_df = pl.DataFrame({"seq": seqs[:60], "rt": rts[:60]})
    val_df = pl.DataFrame({"seq": seqs[60:], "rt": rts[60:]})
    predictor = RTPredictor.from_blank()
    result = predictor.fit(
        train_df, val_df=val_df, epochs=2, batch_size=16, warmup_epochs=0,
        lr=1e-3, seed=0, verbose=False,
    )
    assert np.isfinite(result.best_val_r2)
    assert len(result.history) == 2
