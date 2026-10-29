"""Tokenizer round-trips for the three supported peptide formats. Every
peptide carries a terminus token at both ends: its N-terminal mod or <nterm>,
and <cterm>."""

from jrif.dataset import DEFAULT_TOKENS, MAX_LEN, Vocabulary, tokenize


def _ids_to_tokens(ids, vocab):
    rev = {v: k for k, v in vocab.tokens.items()}
    return [rev[i] for i in ids if i != 0]


def _psmtag_vocab():
    v = Vocabulary.default()
    v.extend(".[308]", None, fresh=True)
    return v


def test_bracket_format_roundtrip():
    vocab = Vocabulary.default()
    # Bare C in bracket format means camC (C[160])
    assert _ids_to_tokens(tokenize("AAMCK", vocab), vocab) == \
        ["<nterm>", "A", "A", "M", "C[160]", "K", "<cterm>"]


def test_bracket_format_with_oxm():
    vocab = Vocabulary.default()
    assert _ids_to_tokens(tokenize("M[147]AAR", vocab), vocab) == \
        ["<nterm>", "M[147]", "A", "A", "R", "<cterm>"]


def test_unimod_format_oxm_and_camc():
    vocab = Vocabulary.default()
    assert _ids_to_tokens(tokenize("AM(UniMod:35)C(UniMod:4)R", vocab), vocab) == \
        ["<nterm>", "A", "M[147]", "C[160]", "R", "<cterm>"]


def test_psmtag_format_full():
    vocab = _psmtag_vocab()
    tokens = tokenize(".[308.1161]AK[308.1161]C(UniMod:4)R", vocab)
    # the N-terminal mod takes the <nterm> slot
    assert _ids_to_tokens(tokens, vocab) == [".[308]", "A", "K[308]", "C[160]", "R", "<cterm>"]


def test_psmtag_oxm():
    vocab = _psmtag_vocab()
    tokens = tokenize(".[308.1161]AM(UniMod:35)K[308.1161]R", vocab)
    assert _ids_to_tokens(tokens, vocab) == [".[308]", "A", "M[147]", "K[308]", "R", "<cterm>"]


def test_padding_and_max_len():
    vocab = Vocabulary.default()
    tokens = tokenize("A" * (MAX_LEN + 5), vocab)
    assert len(tokens) == MAX_LEN
    assert tokens[0] == DEFAULT_TOKENS["<nterm>"]
    assert all(t == DEFAULT_TOKENS["A"] for t in tokens[1:])


def test_short_sequence_zero_padded():
    vocab = Vocabulary.default()
    tokens = tokenize("AAA", vocab)
    assert len(tokens) == MAX_LEN
    assert tokens[:5] == [DEFAULT_TOKENS["<nterm>"]] + [DEFAULT_TOKENS["A"]] * 3 \
        + [DEFAULT_TOKENS["<cterm>"]]
    assert tokens[5:] == [0] * (MAX_LEN - 5)


def test_empty_sequence():
    assert tokenize("", Vocabulary.default()) == [0] * MAX_LEN
