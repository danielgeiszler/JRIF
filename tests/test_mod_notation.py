"""One modification notation: `<site>[<signed delta>]` at
the precision written; known chemistry at 4 decimals; written mods match
vocabulary tokens at the coarser precision, never ambiguously."""

import numpy as np
import polars as pl
import pytest

from jrif import IMPredictor, RTPredictor, TokenizationError
from jrif.checkpoints import checkpoint_path
from jrif.dataset import UNIMOD_DELTA, Vocabulary, canonical_token, tokenize


def _names(seq, vocab, **kw):
    rev = {i: t for t, i in vocab.tokens.items()}
    return [rev[i] for i in tokenize(seq, vocab, **kw) if i != 0]


@pytest.mark.parametrize("spellings,want", [
    # legacy aliases (the shipped models' established spellings) and UniMod
    (["C[160]", "C[160.0307]", "C(UniMod:4)", "C[+57.0215]"], "C[+57.0215]"),
    (["M[147]", "M[147.0354]", "M(UniMod:35)", "M[+15.9949]"], "M[+15.9949]"),
    (["K[308]", "K[308.1161]", "K[+308.1161]"], "K[+308.1161]"),
    ([".[308]", ".[308.1161]", "n[+308.1161]", "[+308.1161]-"], "n[+308.1161]"),
    # everything else: the delta exactly as written, no added precision
    (["S[80]", "S[+80]", "S[80.]"], "S[+80]"),
    (["S(UniMod:21)", "S[+79.9663]", "S[79.9663]"], "S[+79.9663]"),
    (["(UniMod:1)", "[+42.0106]-", ".[42.0106]"], "n[+42.0106]"),
    (["Q(UniMod:28)", "Q[-17.0265]"], "Q[-17.0265]"),
])
def test_equivalent_spellings_share_one_name(spellings, want):
    assert {canonical_token(s) for s in spellings} == {want}


def test_precision_is_kept_as_written():
    assert canonical_token("S[+123.45678]") == "S[+123.45678]"
    assert canonical_token("K[-0.5]") == "K[-0.5]"
    assert canonical_token("S[80.0]") == "S[+80.0]"
    # legacy aliases are per-site: 160 on S is just +160
    assert canonical_token("S[160]") == "S[+160]"
    # a signed mass is never aliased
    assert canonical_token("K[+308]") == "K[+308]"


def test_written_mod_matches_at_coarser_precision():
    v = Vocabulary.default()
    v.extend("S[+79.9663]", "S")
    for spelling in ("S[80]", "S[+80]", "S[+80.0]", "S[+79.97]", "S(UniMod:21)"):
        assert v.match(canonical_token(spelling)) == "S[+79.9663]", spelling
    assert v.match(canonical_token("S[+79.95]")) is None
    # signed integer tag mass resolves to the shipped tag token
    assert Vocabulary.default().match(canonical_token("K[+308]")) == "K[+308.1161]"
    # half-up rounding, not banker's: 80.5 is 81 at integer precision
    v.extend("T[+81]", "T")
    assert v.match(canonical_token("T[+80.5]")) == "T[+81]"


def test_ambiguous_match_raises():
    v = Vocabulary.default()
    v.extend("S[+79.9663]", "S")
    v.extend("S[+79.95]", "S")
    with pytest.raises(TokenizationError, match=r"^'S\[80\]' in 'PES\[80\]K' matches several"):
        tokenize("PES[80]K", v)


def test_vocabulary_refuses_conflicting_tokens():
    v = Vocabulary.default()
    v.extend("S[+79.9663]", "S")
    with pytest.raises(ValueError, match="agrees with existing"):
        v.extend("S[+80]", "S")
    with pytest.raises(ValueError, match="not a token name"):
        v.extend(".[308]", None)


def test_agreeing_spellings_become_one_new_token_named_most_precisely():
    from jrif.dataset import discover_tokens
    new = discover_tokens(["AT[80]K", "AT[+79.9663]K", "AS[80]K"], Vocabulary.default())
    assert new == {"T[+79.9663]": "T", "S[+80]": "S"}


def test_unimod_any_casing_full_table():
    assert len(UNIMOD_DELTA) > 1000
    for spelling in ("S(UniMod:30)", "S(Unimod:30)", "S(unimod:30)"):
        assert canonical_token(spelling) == "S[+21.9819]"  # Cation:Na


def test_unknown_unimod_raises():
    with pytest.raises(TokenizationError, match="UniMod:99999"):
        tokenize("PEPS(UniMod:99999)K", Vocabulary.default())


def test_unparseable_character_raises():
    for seq in ("PEP_TIDE", "PEPtide", "PEP*TIDE"):
        with pytest.raises(TokenizationError, match="unparseable"):
            tokenize(seq, Vocabulary.default())


def test_errors_quote_the_spelling_as_written():
    with pytest.raises(TokenizationError, match=r"'Y\[80\.0\]'"):
        tokenize("PEPY[80.0]K", Vocabulary.default())


def test_bare_cysteine_is_unmodified_in_every_notation():
    v = Vocabulary.default()
    v.extend("n[+308.1161]", None, fresh=True)
    for seq in ("ACK", ".[308]ACK", "M(UniMod:35)ACK"):
        names = _names(seq, v)
        assert "C" in names and "C[+57.0215]" not in names


def test_cterm_mod_takes_the_cterm_slot():
    v = Vocabulary.default()
    v.extend("c[+0.984]", None)
    names = _names("PEPTIDEc[+0.984]", v)
    assert names[0] == "<nterm>" and names[-1] == "c[+0.984]" and "<cterm>" not in names


@pytest.mark.parametrize("name", ["rt_lf", "rt_psmtag"])
def test_shipped_rt_checkpoints_use_the_notation(name):
    v = RTPredictor.from_checkpoint(checkpoint_path(name)).vocab
    assert {"C[+57.0215]", "M[+15.9949]", "K[+308.1161]"} <= set(v.tokens)
    assert not any(t in v.tokens for t in ("C[160]", "M[147]", "K[308]", ".[308]"))


def test_old_and_new_spellings_predict_identically():
    p = RTPredictor.from_checkpoint(checkpoint_path("rt_psmtag"))
    old = pl.DataFrame({"seq": [".[308]PEPC[160]K[308]", ".[308]AM[147]K[308]"]})
    new = pl.DataFrame({"seq": ["n[+308.1161]PEPC(UniMod:4)K[+308.1161]",
                                "n[+308.1161]AM(UniMod:35)K[+308.1161]"]})
    np.testing.assert_allclose(p.predict(old, device="cpu"), p.predict(new, device="cpu"))
    m = IMPredictor.from_checkpoint(checkpoint_path("im_psmtag"))
    np.testing.assert_allclose(m.predict(old.with_columns(charge=pl.lit(2)), device="cpu"),
                               m.predict(new.with_columns(charge=pl.lit(2)), device="cpu"))
