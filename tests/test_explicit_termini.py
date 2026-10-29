"""Terminus tokens: every peptide carries a token at both ends — its
N-terminal mod or <nterm>, and <cterm>."""

import numpy as np
import polars as pl

from jrif import IMPredictor, RTPredictor
from jrif.checkpoints import checkpoint_path
from jrif.dataset import FREE_CTERM, FREE_NTERM, Vocabulary, tokenize


def _nonpad(ids):
    return [i for i in ids if i != 0]


def test_free_termini_wrap_unmodified_peptide():
    v = Vocabulary.default()
    ids = _nonpad(tokenize("PEPTIDEK", v, strict=True))
    assert len(ids) == 10
    assert ids[0] == v.tokens[FREE_NTERM] and ids[-1] == v.tokens[FREE_CTERM]


def test_nterm_mod_replaces_free_nterm():
    v = Vocabulary.default()
    v.extend(".[308]", None, fresh=True)
    ids = _nonpad(tokenize(".[308]PEPTIDEK[308]", v, strict=True))
    assert ids[0] == v.tokens[".[308]"] and ids[-1] == v.tokens[FREE_CTERM]
    assert v.tokens[FREE_NTERM] not in ids
    assert len(ids) == 10


def test_shipped_checkpoints_load_with_terminus_tokens():
    for name in ("rt_lf", "rt_psmtag"):
        v = RTPredictor.from_checkpoint(checkpoint_path(name)).vocab
        assert FREE_NTERM in v.tokens and FREE_CTERM in v.tokens
    v = IMPredictor.from_checkpoint(checkpoint_path("im_psmtag")).vocab
    assert FREE_NTERM in v.tokens and FREE_CTERM in v.tokens


def test_fit_predict_roundtrip_with_tagged_and_free_termini(tmp_path):
    p = RTPredictor.from_blank()
    seqs = [f"PEPTIDEK{aa}" for aa in "ACDEFGHILMNQRSTVWY"]
    seqs = seqs + [".[308]" + s for s in seqs]
    df = pl.DataFrame({"seq": seqs, "rt": list(np.linspace(0.0, 60.0, len(seqs)))})
    p.fit(df, epochs=1, verbose=False)
    assert ".[308]" in p.vocab.tokens
    p.save(tmp_path / "m.pt")
    q = RTPredictor.from_checkpoint(tmp_path / "m.pt")
    assert q.vocab.tokens == p.vocab.tokens
    np.testing.assert_allclose(q.predict(df, device="cpu"), p.predict(df, device="cpu"),
                               rtol=1e-5, atol=1e-5)


def test_im_fit_predict():
    m = IMPredictor.from_blank()
    seqs = [f"PEPTIDEK{aa}" for aa in "ACDEFGHILMNQRSTVWY"] * 2
    df = pl.DataFrame({"seq": seqs, "charge": [2, 3] * 18,
                       "im": list(np.linspace(0.7, 1.3, len(seqs)))})
    m.fit(df, epochs=1, verbose=False)
    assert np.all(np.isfinite(m.predict(df)))
