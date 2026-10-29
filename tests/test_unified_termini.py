"""Termini live only in the token sequence: model mechanics, checkpoint
round-trip, and calibrate() behavior (warp-first, fallback, curriculum)."""

import numpy as np
import pandas as pd
import pytest
import torch

from jrif.api import CHECKPOINT_FORMAT_VERSION, RTPredictor
from jrif.dataset import Vocabulary, tokenize
from jrif.model import RTModel


def _psmtag_vocab():
    v = Vocabulary.default()
    v.extend("n[+308.1161]", None, fresh=True)
    return v


def test_model_head_sees_only_pooled_conv_and_globals():
    m = RTModel(Vocabulary.default())
    assert m.head[1].in_features == 133
    assert {n.split(".")[0] for n, _ in m.named_parameters()} == {"embedding", "convs", "head"}


def test_extend_vocab_does_not_perturb_existing_predictions():
    m = RTModel(_psmtag_vocab())
    m.eval()
    x = torch.tensor([tokenize(".[308]ACK[308]", m.vocab)])
    with torch.no_grad():
        before = m(x).clone()
    m.extend_vocab([("S[+80]", "S", False), ("n[+999]", None, True)])
    m.eval()
    with torch.no_grad():
        after = m(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_fit_predict_checkpoint_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    seqs = [".[308]" + "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 8)) + "K[308]"
            for _ in range(200)]
    df = pd.DataFrame({"seq": seqs, "rt": rng.normal(50, 10, 200)})

    p = RTPredictor.from_blank()
    p.fit(df, epochs=2, device="cpu", verbose=False)
    preds = p.predict(df, device="cpu")
    assert preds.shape == (200,)
    assert "n[+308.1161]" in p.vocab.tokens  # discovered and learned

    path = tmp_path / "rt.pt"
    p.save(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert ckpt["format_version"] == CHECKPOINT_FORMAT_VERSION

    q = RTPredictor.from_checkpoint(path)
    preds2 = q.predict(df, device="cpu")
    assert np.allclose(preds, preds2, atol=1e-5)


def test_calibrate_isotonic_and_roundtrip(tmp_path):
    """Matched-vocabulary small-N -> isotonic mode; warp survives save/load."""
    rng = np.random.default_rng(0)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    w = dict(zip(aas, rng.normal(0, 3, 20)))  # learnable additive chemistry
    seqs = ["".join(rng.choice(aas, rng.integers(7, 20))) for _ in range(400)]
    df = pd.DataFrame({"seq": seqs,
                       "rt": [50 + sum(w[a] for a in s) for s in seqs]})
    p = RTPredictor.from_blank()
    p.fit(df, epochs=60, lr=1e-3, device="cpu", verbose=False)
    # new experiment: nonlinear monotone warp of the same chemistry
    warped = df.assign(rt=np.exp(df["rt"] / 25.0).clip(upper=1e6))
    cal = warped.iloc[:300]
    rep = p.calibrate(cal, device="cpu", verbose=False, epochs=2)
    assert rep["mode"] in ("warp", "warp+finetune")
    assert rep["holdout_rank"] > 0.95
    preds = p.predict(warped.iloc[300:], device="cpu")
    assert preds.min() > 0  # output is in warped units, not z
    path = tmp_path / "cal.pt"
    p.save(path)
    q = RTPredictor.from_checkpoint(path)
    assert q.rt_warp is not None
    assert np.allclose(q.predict(warped.iloc[300:], device="cpu"), preds, atol=1e-5)


def test_calibrate_falls_back_to_finetune_on_broken_rank():
    rng = np.random.default_rng(1)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    seqs = ["".join(rng.choice(aas, rng.integers(7, 20))) for _ in range(200)]
    df = pd.DataFrame({"seq": seqs, "rt": rng.normal(0, 1, 200)})  # random targets
    p = RTPredictor.from_blank()
    rep = p.calibrate(df, device="cpu", verbose=False, epochs=2)
    assert rep["mode"] == "finetune_raw"
    assert p.rt_warp is None


def test_calibrate_curriculum_with_novel_tokens(tmp_path):
    rng = np.random.default_rng(0)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    w = dict(zip(aas, rng.normal(0, 3, 20)))
    seqs = ["".join(rng.choice(aas, rng.integers(7, 20))) for _ in range(400)]
    df = pd.DataFrame({"seq": seqs,
                       "rt": [50 + sum(w[a] for a in s) for s in seqs]})
    p = RTPredictor.from_blank()
    p.fit(df, epochs=60, lr=1e-3, device="cpu", verbose=False)
    # new experiment: warped scale, and a novel mod on ~25% of peptides
    new = df.assign(rt=np.exp(df["rt"] / 25.0).clip(upper=1e6))
    mod = rng.random(len(new)) < 0.25
    new.loc[mod, "seq"] = new.loc[mod, "seq"].str.replace("A", "A[42]", n=1)
    rep = p.calibrate(new, device="cpu", verbose=False, curriculum=True, epochs=2)
    assert rep["mode"] in ("curriculum", "curriculum_raw")
    assert "A[+42]" in p.vocab.tokens
    preds = p.predict(new.head(20), device="cpu")
    assert np.isfinite(preds).all()


def test_calibrate_curriculum_no_novel_tokens_falls_through():
    rng = np.random.default_rng(1)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    seqs = ["".join(rng.choice(aas, rng.integers(7, 20))) for _ in range(150)]
    df = pd.DataFrame({"seq": seqs, "rt": rng.normal(0, 1, 150)})
    p = RTPredictor.from_blank()
    rep = p.calibrate(df, device="cpu", verbose=False, curriculum=True, epochs=2)
    assert rep["mode"] == "finetune_raw"   # unrankable -> single-phase fallback


def test_calibrate_with_novel_mod_learns_it_in_every_mode():
    """New tokens are added before the base snapshot, so restoring the base
    weights (warp-only) works and the new mod is predictable either way."""
    rng = np.random.default_rng(0)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    w = dict(zip(aas, rng.normal(0, 3, 20)))
    seqs = ["".join(rng.choice(aas, rng.integers(7, 20))) for _ in range(400)]
    df = pd.DataFrame({"seq": seqs, "rt": [50 + sum(w[a] for a in s) for s in seqs]})
    p = RTPredictor.from_blank()
    p.fit(df, epochs=60, lr=1e-3, device="cpu", verbose=False)
    new = df.assign(rt=np.exp(df["rt"] / 25.0).clip(upper=1e6))
    mod = rng.random(len(new)) < 0.25
    new.loc[mod, "seq"] = new.loc[mod, "seq"].str.replace("A", "A[42]", n=1)
    # lr=5 wrecks fine-tuning, so warp-only wins and the base weights are restored
    for lr, modes in ((5.0, {"warp"}), (1e-3, {"warp", "warp+finetune"})):
        q = RTPredictor.from_blank()
        q.model.load_state_dict(p.model.state_dict())
        q.rt_median, q.rt_mad = p.rt_median, p.rt_mad
        rep = q.calibrate(new, device="cpu", verbose=False, epochs=2, lr=lr)
        assert rep["mode"] in modes
        assert "A[+42]" in q.vocab.tokens
        assert np.isfinite(q.predict(new[mod].head(10), device="cpu")).all()
