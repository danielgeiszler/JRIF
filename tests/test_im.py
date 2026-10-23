"""IMModel / IMPredictor: charge-as-input-channel ion-mobility prediction."""

import numpy as np
import pandas as pd
import pytest
import torch

from jrif.im import IMDataset, IMModel, IMPredictor
from jrif.dataset import Vocabulary


def _df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    w = dict(zip(aas, rng.normal(0, 0.01, 20)))
    seqs = ["".join(rng.choice(aas, rng.integers(7, 20))) for _ in range(n)]
    charges = rng.choice([2, 3], n)
    im = np.array([0.7 + 0.15 * (c - 2) + sum(w[a] for a in s)
                   for s, c in zip(seqs, charges)])
    return pd.DataFrame({"seq": seqs, "charge": charges, "im": im})


def test_charge_changes_prediction():
    m = IMModel(Vocabulary.default())
    m.eval()
    v = m.vocab
    x = torch.tensor([[v.tokens["A"], v.tokens["C[160]"], v.tokens["K"]] + [0] * 47])
    with torch.no_grad():
        p2 = m(x, torch.tensor([0.0]))
        p3 = m(x, torch.tensor([1.0]))
    assert not torch.allclose(p2, p3)


def test_fit_predict_roundtrip(tmp_path):
    df = _df()
    p = IMPredictor.from_blank()
    p.fit(df, epochs=3, device="cpu", verbose=False)
    preds = p.predict(df, device="cpu")
    assert preds.shape == (len(df),)
    path = tmp_path / "im.pt"
    p.save(path)
    q = IMPredictor.from_checkpoint(path)
    assert np.allclose(q.predict(df, device="cpu"), preds, atol=1e-5)


def test_im_checkpoint_kind_guard(tmp_path):
    df = _df(100)
    p = IMPredictor.from_blank()
    p.fit(df, epochs=1, device="cpu", verbose=False)
    path = tmp_path / "im.pt"
    p.save(path)
    from jrif.api import RTPredictor
    rt = RTPredictor.from_blank(arch="dilated_cnn", unified_termini=True)
    rt_path = tmp_path / "rt.pt"
    rt.save(rt_path)
    with pytest.raises(ValueError):
        IMPredictor.from_checkpoint(rt_path)


def test_fit_discovers_new_tokens():
    df = _df(150)
    df["seq"] = ".[308]" + df["seq"]
    p = IMPredictor.from_blank()
    p.fit(df, epochs=1, device="cpu", verbose=False)
    assert ".[308]" in p.vocab.tokens
