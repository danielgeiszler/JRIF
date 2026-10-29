import numpy as np
import polars as pl
import pytest

from jrif import IMPredictor, RTPredictor, TokenizationError
from jrif.dataset import MAX_LEN, Vocabulary, tokenize


VOCAB = Vocabulary.default()


def test_strict_passes_on_known_sequence():
    t = tokenize("PEPTIDEK", VOCAB, strict=True)
    assert sum(1 for x in t if x != 0) == 10  # 8 residues + both termini


def test_strict_raises_on_unknown_mod():
    tokenize("PES[80]TIDEK", VOCAB)  # lenient: falls back to bare S
    with pytest.raises(TokenizationError, match=r"S\[80\]"):
        tokenize("PES[80]TIDEK", VOCAB, strict=True)


def test_strict_raises_on_unknown_nterm_mod():
    with pytest.raises(TokenizationError, match="N-terminal"):
        tokenize(".[999]PEPTIDEK", VOCAB, strict=True)


def test_strict_raises_on_empty_or_non_string():
    with pytest.raises(TokenizationError):
        tokenize("", VOCAB, strict=True)
    with pytest.raises(TokenizationError):
        tokenize(None, VOCAB, strict=True)


def test_strict_raises_on_truncation():
    seq = "A" * (MAX_LEN + 1)
    t = tokenize(seq, VOCAB)  # lenient: truncated
    assert len(t) == MAX_LEN
    with pytest.raises(TokenizationError, match="MAX_LEN"):
        tokenize(seq, VOCAB, strict=True)


def test_rt_predict_strict_default_raises_and_opt_out_works():
    p = RTPredictor.from_blank()
    lib = pl.DataFrame({"seq": ["PEPTIDEK", "PES[80]TIDEK"]})
    with pytest.raises(TokenizationError):
        p.predict(lib)
    out = p.predict(lib, strict=False)
    assert out.shape == (2,) and np.all(np.isfinite(out))


def test_fit_skips_untokenizable_rows_with_one_log_line(capsys):
    p = RTPredictor.from_blank()
    seqs = [f"PEPTIDEK{aa}" for aa in "ACDEFGHILMNQRSTVWY"] * 3
    rts = list(np.linspace(0.0, 60.0, len(seqs)))
    bad = "A" * (MAX_LEN + 5)
    df = pl.DataFrame({"seq": seqs + [bad], "rt": rts + [30.0]})
    p.fit(df, epochs=1, verbose=False)
    out = capsys.readouterr().out
    assert out.count("jrif: skipped") == 1 and "MAX_LEN" in out


def test_fit_logs_nothing_when_all_rows_tokenize(capsys):
    p = RTPredictor.from_blank()
    seqs = [f"PEPTIDEK{aa}" for aa in "ACDEFGHILMNQRSTVWY"] * 3
    df = pl.DataFrame({"seq": seqs, "rt": list(np.linspace(0.0, 60.0, len(seqs)))})
    p.fit(df, epochs=1, verbose=False)
    assert "jrif: skipped" not in capsys.readouterr().out


def test_fit_still_learns_novel_mods_without_skipping(capsys):
    p = RTPredictor.from_blank()
    seqs = [f"PES[80]TIDEK{aa}" for aa in "ACDEFGHILMNQRSTVWY"] * 3
    df = pl.DataFrame({"seq": seqs, "rt": list(np.linspace(0.0, 60.0, len(seqs)))})
    p.fit(df, epochs=1, verbose=False)
    assert "jrif: skipped" not in capsys.readouterr().out
    assert "S[80]" in p.vocab.tokens


def test_calibrate_skips_untokenizable_rows(capsys):
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=12)) for _ in range(120)]
    rts = rng.uniform(0, 60, size=len(seqs))
    bad = "A" * (MAX_LEN + 5)
    df = pl.DataFrame({"seq": seqs + [bad], "rt": list(rts) + [30.0]})
    p = RTPredictor.from_blank()
    report = p.calibrate(df, epochs=1, verbose=False)
    out = capsys.readouterr().out
    assert out.count("jrif: skipped 1") == 1
    assert report["n"] == 120


def test_im_predict_strict_default_raises_and_opt_out_works():
    m = IMPredictor.from_blank()
    lib = pl.DataFrame({"seq": ["PEPTIDEK", "PES[80]TIDEK"], "charge": [2, 3]})
    with pytest.raises(TokenizationError):
        m.predict(lib)
    out = m.predict(lib, strict=False)
    assert out.shape == (2,) and np.all(np.isfinite(out))
