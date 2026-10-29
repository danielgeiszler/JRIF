"""Vocab extension preserves predictions on peptides that don't contain the
new tokens. This guards against the K[+308.1161]-style regression where adding a
new token silently changed behavior on unrelated inputs."""

import polars as pl
import torch

from jrif.dataset import RTDataset, Vocabulary, discover_tokens
from jrif.model import RTModel


def _predict(model, ds):
    model.eval()
    with torch.no_grad():
        return model(ds.sequences).cpu().clone()


def _extend_and_compare(new_tokens, seqs):
    torch.manual_seed(0)
    vocab = Vocabulary.default()
    model = RTModel(vocab, embedding_dim=16)
    base_df = pl.DataFrame({"seq": seqs, "rt": [0.1] * len(seqs)})
    before = _predict(model, RTDataset(base_df, vocab, rt_median=0.0, rt_mad=1.0))
    model.extend_vocab(new_tokens)
    assert model.embedding.num_embeddings == vocab.size
    # Rebuild the dataset AFTER extension (so vocab is consistent); none of
    # the peptides contain the new token, so predictions must be unchanged.
    after = _predict(model, RTDataset(base_df, vocab, rt_median=0.0, rt_mad=1.0))
    assert torch.allclose(before, after, atol=1e-6)
    return model


def test_extend_inherit_keeps_base_predictions():
    model = _extend_and_compare([("S[+80]", "S", False)], ["AAAR", "MCAR", "PEPTIDEK"])
    assert "S[+80]" in model.vocab.tokens


def test_extend_fresh_keeps_base_predictions():
    model = _extend_and_compare([("Y[+42]", None, True)], ["AAAR", "MCAR"])
    assert "Y[+42]" in model.vocab.fresh


def test_extend_nterm_mod_keeps_base_predictions():
    model = _extend_and_compare([("n[+42]", None, True)], ["AAAR", "MCAR"])
    assert "n[+42]" in model.vocab.tokens


def test_discover_tokens_finds_unknown():
    vocab = Vocabulary.default()
    new = discover_tokens(["AAAR", "AS[80]K", "PEPTIDET[80]K", "M(UniMod:35)AR"], vocab)
    assert new == {"S[+80]": "S", "T[+80]": "T"}


def test_discover_tokens_finds_nterm_mod():
    vocab = Vocabulary.default()
    new = discover_tokens([".[42.0]AAR", ".[308.1161]MCAR"], vocab)
    assert new == {"n[+42.0]": "<nterm>", "n[+308.1161]": "<nterm>"}
    vocab.extend("n[+308.1161]", None, fresh=True)
    assert not discover_tokens([".[308.1161]MCAR", ".[308]MCAR"], vocab)


def test_new_mods_start_as_their_unmodified_form():
    """Before any training, a peptide carrying a new mod predicts exactly like
    its unmodified form: residue mods copy their residue, terminal mods copy
    <nterm>/<cterm>."""
    from jrif.api import _extend_for
    torch.manual_seed(0)
    model = RTModel(Vocabulary.default(), embedding_dim=16)
    mods = ["n[+140.0950]PEPTIDEK", "PEPTIDEKc[+0.984]", "PEPS[80]TIDEK"]
    plain = ["PEPTIDEK", "PEPTIDEK", "PEPSTIDEK"]
    _extend_for(model, mods)
    df = lambda seqs: pl.DataFrame({"seq": seqs, "rt": [0.0] * len(seqs)})
    v = model.vocab
    assert v.parent_of["n[+140.0950]"] == "<nterm>" and v.parent_of["c[+0.984]"] == "<cterm>"
    a = _predict(model, RTDataset(df(mods), v, rt_median=0.0, rt_mad=1.0))
    b = _predict(model, RTDataset(df(plain), v, rt_median=0.0, rt_mad=1.0))
    assert torch.allclose(a, b, atol=1e-6)
