"""Tokenizer round-trips for the three supported peptide formats."""

from jrif.dataset import (
    DEFAULT_NTERM_MOD_TOKENS, DEFAULT_TOKENS, MAX_LEN, Vocabulary, tokenize,
)


def _ids_to_tokens(ids, vocab):
    rev = {v: k for k, v in vocab.tokens.items()}
    return [rev[i] for i in ids if i != 0]


def test_bracket_format_roundtrip():
    vocab = Vocabulary.default()
    tokens, n_id, c_id = tokenize("AAMCK", vocab)
    # Bare C in bracket format means camC (C[160])
    assert _ids_to_tokens(tokens, vocab) == ["A", "A", "M", "C[160]", "K"]
    assert n_id == 0
    assert c_id == 0


def test_bracket_format_with_oxm():
    vocab = Vocabulary.default()
    tokens, n_id, _ = tokenize("M[147]AAR", vocab)
    assert _ids_to_tokens(tokens, vocab) == ["M[147]", "A", "A", "R"]
    assert n_id == 0


def test_unimod_format_oxm_and_camc():
    vocab = Vocabulary.default()
    tokens, _, _ = tokenize("AM(UniMod:35)C(UniMod:4)R", vocab)
    assert _ids_to_tokens(tokens, vocab) == ["A", "M[147]", "C[160]", "R"]


def test_psmtag_format_full():
    vocab = Vocabulary.default()
    tokens, n_id, _ = tokenize(".[308.1161]AK[308.1161]C(UniMod:4)R", vocab)
    assert _ids_to_tokens(tokens, vocab) == ["A", "K[308]", "C[160]", "R"]
    assert n_id == DEFAULT_NTERM_MOD_TOKENS[".[308]"]


def test_psmtag_oxm():
    vocab = Vocabulary.default()
    tokens, _, _ = tokenize(".[308.1161]AM(UniMod:35)K[308.1161]R", vocab)
    assert _ids_to_tokens(tokens, vocab) == ["A", "M[147]", "K[308]", "R"]


def test_padding_and_max_len():
    vocab = Vocabulary.default()
    long_seq = "A" * (MAX_LEN + 5)
    tokens, _, _ = tokenize(long_seq, vocab)
    assert len(tokens) == MAX_LEN
    assert all(t == DEFAULT_TOKENS["A"] for t in tokens)


def test_short_sequence_zero_padded():
    vocab = Vocabulary.default()
    tokens, _, _ = tokenize("AAA", vocab)
    assert len(tokens) == MAX_LEN
    assert tokens[:3] == [DEFAULT_TOKENS["A"]] * 3
    assert tokens[3:] == [0] * (MAX_LEN - 3)


def test_empty_sequence():
    vocab = Vocabulary.default()
    tokens, n_id, c_id = tokenize("", vocab)
    assert tokens == [0] * MAX_LEN
    assert n_id == 0 and c_id == 0
