import numpy as np
import polars as pl
import pytest

from jrif import IMPredictor, RTPredictor, TokenizationError
from jrif.dataset import MAX_LEN, Vocabulary, tokenize


VOCAB = Vocabulary.default()


def test_known_sequence_tokenizes():
    t = tokenize("PEPTIDEK", VOCAB)
    assert sum(1 for x in t if x != 0) == 10  # 8 residues + both termini


def test_unknown_mod_raises():
    with pytest.raises(TokenizationError, match=r"S\[80\]"):
        tokenize("PES[80]TIDEK", VOCAB)


def test_unknown_nterm_mod_raises():
    with pytest.raises(TokenizationError, match="N-terminal"):
        tokenize(".[999]PEPTIDEK", VOCAB)


def test_empty_or_non_string_raises():
    with pytest.raises(TokenizationError):
        tokenize("", VOCAB)
    with pytest.raises(TokenizationError):
        tokenize(None, VOCAB)


def test_over_length_raises():
    with pytest.raises(TokenizationError, match="MAX_LEN"):
        tokenize("A" * (MAX_LEN + 1), VOCAB)


def test_rt_predict_raises_on_unrepresentable_rows():
    p = RTPredictor.from_blank()
    with pytest.raises(TokenizationError):
        p.predict(pl.DataFrame({"seq": ["PEPTIDEK", "PES[80]TIDEK"]}))


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
    assert "S[+80]" in p.vocab.tokens


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


def test_im_predict_raises_on_unrepresentable_rows():
    m = IMPredictor.from_blank()
    with pytest.raises(TokenizationError):
        m.predict(pl.DataFrame({"seq": ["PEPTIDEK", "PES[80]TIDEK"], "charge": [2, 3]}))


def test_untokenizable_row_never_empties_validation():
    """Unrepresentable rows are dropped before the train/val split, so with a
    single validation slot no seed can leave validation empty."""
    df = pl.DataFrame({"seq": ["PEPTIDEK", "ACDEFGHK", "A" * (MAX_LEN + 5)],
                       "rt": [10.0, 20.0, 30.0]})
    for seed in range(10):
        p = RTPredictor.from_blank()
        p.fit(df, epochs=1, val_fraction=0.34, seed=seed, device="cpu", verbose=False)
        m = IMPredictor.from_blank()
        m.fit(df.with_columns(charge=pl.lit(2), im=pl.col("rt") / 30), epochs=1,
              val_fraction=0.34, seed=seed, device="cpu", verbose=False)


def test_split_is_reproducible():
    from jrif.api import _split_train_val
    df = pl.DataFrame({"seq": [f"PEPTIDE{a}{b}K" for a in "ACDEFGH" for b in "LMNPQRS"],
                       "rt": np.arange(49.0)})
    first = _split_train_val(df, val_fraction=0.2, seed=3)[1]["seq"].to_list()
    for _ in range(20):
        assert _split_train_val(df, val_fraction=0.2, seed=3)[1]["seq"].to_list() == first


def test_fit_with_nothing_to_validate_raises():
    df = pl.DataFrame({"seq": ["PEPTIDEK", "A" * (MAX_LEN + 5)], "rt": [10.0, 30.0]})
    with pytest.raises(ValueError, match="at least one training and one validation"):
        RTPredictor.from_blank().fit(df, epochs=1, device="cpu", verbose=False)
