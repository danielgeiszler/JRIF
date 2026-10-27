"""Unified-termini mode: terminal mods live only in the token sequence.

Validated by bench_termmods_pooling.py (inseq_clean ties the stock model on
the apex PSMtag library, 3 seeds). These tests pin the mechanics: tokenizer
placement, head sizing, discovery routing, checkpoint round-trip, and the
no-perturbation guarantee on vocab extension.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from jrif.api import RTPredictor
from jrif.dataset import RTDataset, Vocabulary, discover_tokens, tokenize
from jrif.model import RTModel


def _unified_vocab():
    v = Vocabulary.default()
    v.extend(".[308]", None, fresh=True)
    return v


def test_tokenize_terminal_in_seq_places_nterm_token_first():
    v = _unified_vocab()
    ids, nterm, cterm = tokenize(".[308]ACK[308]R", v, terminal_in_seq=True)
    assert ids[0] == v.tokens[".[308]"]
    assert ids[1] == v.tokens["A"]
    assert ids[3] == v.tokens["K[308]"]
    assert nterm == 0 and cterm == 0
    # default mode unchanged: mod goes to the separate channel
    ids2, nterm2, _ = tokenize(".[308]ACK[308]R", v)
    assert ids2[0] == v.tokens["A"]
    assert nterm2 == v.nterm_mod_tokens[".[308]"]


def test_discover_routes_nterm_mods_to_main_when_unified():
    v = Vocabulary.default()
    new_main, new_nterm = discover_tokens([".[999]ACK"], v, terminal_in_seq=True)
    assert ".[999]" in new_main and new_main[".[999]"] is None
    assert not new_nterm
    new_main2, new_nterm2 = discover_tokens([".[999]ACK"], v)
    assert ".[999]" in new_nterm2 and ".[999]" not in new_main2


def test_unified_model_has_no_terminal_embeddings_and_small_head():
    m = RTModel(_unified_vocab(), unified_termini=True)
    assert not hasattr(m, "n_term_embedding")
    assert not hasattr(m, "nterm_mod_embedding")
    assert m.head[1].in_features == 133
    legacy = RTModel(Vocabulary.default())
    assert legacy.head[1].in_features == 149


def test_unified_extend_vocab_rejects_terminal_mod_channels():
    m = RTModel(_unified_vocab(), unified_termini=True)
    with pytest.raises(ValueError):
        m.extend_vocab(new_nterm_mods=[".[999]"])


def test_unified_extend_vocab_does_not_perturb_existing_predictions():
    m = RTModel(_unified_vocab(), unified_termini=True)
    m.eval()
    v = m.vocab
    x = torch.tensor([[v.tokens[".[308]"], v.tokens["A"], v.tokens["C[160]"],
                       v.tokens["K[308]"]] + [0] * 46])
    with torch.no_grad():
        before = m(x).clone()
    m.extend_vocab(new_main_tokens=[("S[80]", "S", False), (".[999]", None, True)])
    m.eval()
    with torch.no_grad():
        after = m(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_unified_fit_predict_checkpoint_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    seqs = [".[308]" + "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 8)) + "K[308]"
            for _ in range(200)]
    df = pd.DataFrame({"seq": seqs, "rt": rng.normal(50, 10, 200)})

    p = RTPredictor.from_blank(arch="dilated_cnn", unified_termini=True)
    p.fit(df, epochs=2, device="cpu", verbose=False)
    preds = p.predict(df, device="cpu")
    assert preds.shape == (200,)
    # .[308] was discovered as a MAIN token, not an nterm mod
    assert ".[308]" in p.vocab.tokens

    path = tmp_path / "unified.pt"
    p.save(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["format_version"] == 2 and ckpt["unified_termini"] is True

    q = RTPredictor.from_checkpoint(path)
    assert q.model.unified_termini
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
    p = RTPredictor.from_blank(arch="dilated_cnn", unified_termini=True)
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
    p = RTPredictor.from_blank(arch="dilated_cnn", unified_termini=True)
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
    p = RTPredictor.from_blank(arch="dilated_cnn", unified_termini=True)
    p.fit(df, epochs=60, lr=1e-3, device="cpu", verbose=False)
    # new experiment: warped scale, and a novel mod on ~25% of peptides
    new = df.assign(rt=np.exp(df["rt"] / 25.0).clip(upper=1e6))
    mod = rng.random(len(new)) < 0.25
    new.loc[mod, "seq"] = new.loc[mod, "seq"].str.replace("A", "A[42]", n=1)
    rep = p.calibrate(new, device="cpu", verbose=False, curriculum=True, epochs=2)
    assert rep["mode"] in ("curriculum", "curriculum_raw")
    assert "A[42]" in p.vocab.tokens
    preds = p.predict(new.head(20), device="cpu")
    assert np.isfinite(preds).all()


def test_calibrate_curriculum_no_novel_tokens_falls_through():
    rng = np.random.default_rng(1)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    seqs = ["".join(rng.choice(aas, rng.integers(7, 20))) for _ in range(150)]
    df = pd.DataFrame({"seq": seqs, "rt": rng.normal(0, 1, 150)})
    p = RTPredictor.from_blank(arch="dilated_cnn", unified_termini=True)
    rep = p.calibrate(df, device="cpu", verbose=False, curriculum=True, epochs=2)
    assert rep["mode"] == "finetune_raw"   # unrankable -> single-phase fallback
