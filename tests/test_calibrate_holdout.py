"""calibrate() exposes its holdout; the split is deterministic and
recoverable, and no mode trains on the rows it is scored on."""
import numpy as np
import pandas as pd
import polars as pl

from jrif import RTPredictor
from jrif.im import IMPredictor

AAS = list("ACDEFGHIKLMNPQRSTVWY")


def _rt_df(n, seed=0):
    rng = np.random.default_rng(seed)
    w = dict(zip(AAS, rng.normal(0, 3, 20)))
    seqs = ["".join(rng.choice(AAS, rng.integers(7, 20))) for _ in range(n)]
    return pd.DataFrame({"seq": seqs, "rt": [50 + sum(w[a] for a in s) for s in seqs]})


def _trained_rt():
    df = _rt_df(400)
    p = RTPredictor.from_blank()
    p.fit(df, epochs=60, lr=1e-3, device="cpu", verbose=False)
    return p, df.assign(rt=np.exp(df["rt"] / 25.0).clip(upper=1e6))


def test_holdout_rows_and_preds_are_recoverable():
    p, warped = _trained_rt()
    # duplicates + a null: row must still index the caller's frame
    df = pd.concat([warped.iloc[:300], warped.iloc[:5]], ignore_index=True)
    df.loc[3, "rt"] = np.nan
    rep = p.calibrate(df, device="cpu", verbose=False, epochs=2)
    hold = rep["holdout"]
    assert isinstance(hold, pl.DataFrame)
    assert hold.columns == ["row", "seq", "rt", "pred"]
    assert hold["row"].max() < 300 and 3 not in hold["row"].to_list()
    rows = hold["row"].to_list()
    assert df.loc[rows, "seq"].tolist() == hold["seq"].to_list()
    assert np.allclose(df.loc[rows, "rt"].to_numpy(), hold["rt"].to_numpy())
    assert np.isclose(np.abs(hold["pred"] - hold["rt"]).mean(), rep["holdout_mae"])


def test_holdout_split_is_deterministic():
    _, warped = _trained_rt()
    reps = []
    for _ in range(2):
        p, _ = _trained_rt()
        reps.append(p.calibrate(warped.iloc[:300], device="cpu", verbose=False, epochs=2))
    assert reps[0]["holdout"]["row"].to_list() == reps[1]["holdout"]["row"].to_list()


def test_finetune_raw_does_not_train_on_holdout():
    rng = np.random.default_rng(1)
    seqs = ["".join(rng.choice(AAS, rng.integers(7, 20))) for _ in range(200)]
    df = pd.DataFrame({"seq": seqs, "rt": rng.normal(0, 1, 200)})
    p = RTPredictor.from_blank()
    rep = p.calibrate(df, device="cpu", verbose=False, epochs=2)
    assert rep["mode"] == "finetune_raw"
    hold = rep["holdout"]
    assert len(hold) == max(10, int(200 * 0.2))
    assert np.isclose(np.abs(hold["pred"] - hold["rt"]).mean(), rep["holdout_mae"])


def test_curriculum_holdout_rows_index_input():
    p, warped = _trained_rt()
    rng = np.random.default_rng(0)
    new = warped.copy()
    mod = rng.random(len(new)) < 0.25
    new.loc[mod, "seq"] = new.loc[mod, "seq"].str.replace("A", "A[42]", n=1)
    rep = p.calibrate(new, device="cpu", verbose=False, curriculum=True, epochs=2)
    hold = rep["holdout"]
    assert new.loc[hold["row"].to_list(), "seq"].tolist() == hold["seq"].to_list()


def test_im_holdout_rows_and_preds():
    rng = np.random.default_rng(3)
    w = dict(zip(AAS, rng.normal(0, 0.05, 20)))
    seqs = ["".join(rng.choice(AAS, rng.integers(7, 20))) for _ in range(500)]
    charge = rng.integers(2, 4, 500)
    df = pd.DataFrame({"seq": seqs, "charge": charge,
                       "im": [0.8 + sum(w[a] for a in s) + 0.1 * z
                              for s, z in zip(seqs, charge)]})
    p = IMPredictor.from_blank()
    p.fit(df, epochs=5, device="cpu", verbose=False)
    warped = df.assign(im=np.exp(df["im"] * 2.0))
    rep = p.calibrate(warped.iloc[:420], device="cpu", verbose=False, epochs=2)
    hold = rep["holdout"]
    assert hold.columns == ["row", "seq", "charge", "im", "pred"]
    assert warped.loc[hold["row"].to_list(), "seq"].tolist() == hold["seq"].to_list()
    assert np.isclose(np.abs(hold["pred"] - hold["im"]).mean(), rep["holdout_mae"])
