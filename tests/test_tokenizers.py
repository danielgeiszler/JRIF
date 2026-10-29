"""Tokenizer round-trips for the three supported peptide formats. Every
peptide carries a terminus token at both ends: its N-terminal mod or <nterm>,
and <cterm>."""

import pytest

from jrif.dataset import DEFAULT_TOKENS, MAX_LEN, TokenizationError, Vocabulary, tokenize


def _ids_to_tokens(ids, vocab):
    rev = {v: k for k, v in vocab.tokens.items()}
    return [rev[i] for i in ids if i != 0]


def _psmtag_vocab():
    v = Vocabulary.default()
    v.extend("n[+308.1161]", None, fresh=True)
    return v


def test_bracket_format_roundtrip():
    vocab = Vocabulary.default()
    # Bare residues are unmodified: a bare C is cysteine, not camC
    assert _ids_to_tokens(tokenize("AAMCK", vocab), vocab) == \
        ["<nterm>", "A", "A", "M", "C", "K", "<cterm>"]


def test_bracket_format_with_oxm():
    vocab = Vocabulary.default()
    assert _ids_to_tokens(tokenize("M[147]AAR", vocab), vocab) == \
        ["<nterm>", "M[+15.9949]", "A", "A", "R", "<cterm>"]


def test_unimod_format_oxm_and_camc():
    vocab = Vocabulary.default()
    assert _ids_to_tokens(tokenize("AM(UniMod:35)C(UniMod:4)R", vocab), vocab) == \
        ["<nterm>", "A", "M[+15.9949]", "C[+57.0215]", "R", "<cterm>"]


def test_psmtag_format_full():
    vocab = _psmtag_vocab()
    tokens = tokenize(".[308.1161]AK[308.1161]C(UniMod:4)R", vocab)
    # the N-terminal mod takes the <nterm> slot
    assert _ids_to_tokens(tokens, vocab) == ["n[+308.1161]", "A", "K[+308.1161]", "C[+57.0215]", "R", "<cterm>"]


def test_psmtag_oxm():
    vocab = _psmtag_vocab()
    tokens = tokenize(".[308.1161]AM(UniMod:35)K[308.1161]R", vocab)
    assert _ids_to_tokens(tokens, vocab) == ["n[+308.1161]", "A", "M[+15.9949]", "K[+308.1161]", "R", "<cterm>"]


def test_max_len_counts_both_termini():
    vocab = Vocabulary.default()
    tokens = tokenize("A" * (MAX_LEN - 2), vocab)  # 48 residues + 2 termini
    assert tokens[0] == DEFAULT_TOKENS["<nterm>"] and tokens[-1] == DEFAULT_TOKENS["<cterm>"]
    with pytest.raises(TokenizationError, match="MAX_LEN"):
        tokenize("A" * (MAX_LEN - 1), vocab)


def test_short_sequence_zero_padded():
    vocab = Vocabulary.default()
    tokens = tokenize("AAA", vocab)
    assert len(tokens) == MAX_LEN
    assert tokens[:5] == [DEFAULT_TOKENS["<nterm>"]] + [DEFAULT_TOKENS["A"]] * 3 \
        + [DEFAULT_TOKENS["<cterm>"]]
    assert tokens[5:] == [0] * (MAX_LEN - 5)


def test_empty_sequence_raises():
    with pytest.raises(TokenizationError):
        tokenize("", Vocabulary.default())
