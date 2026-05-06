"""Vocab extension preserves predictions on peptides that don't contain the
new tokens. This guards against the K[308]-style regression where adding a
new token silently changed behavior on unrelated inputs."""

import torch

from jrif.dataset import RTDataset, Vocabulary, discover_tokens
from jrif.model import RTModel


def _predict(model, ds):
    model.eval()
    seqs = ds.sequences
    nterm = ds.nterm_mod_ids
    cterm = ds.cterm_mod_ids
    with torch.no_grad():
        return model(seqs, nterm, cterm).cpu().clone()


def test_extend_inherit_keeps_base_predictions():
    torch.manual_seed(0)
    vocab = Vocabulary.default()
    model = RTModel(vocab, embedding_dim=16)

    import polars as pl
    base_df = pl.DataFrame({"seq": ["AAAR", "MCAR", "PEPTIDEK"], "rt": [0.1, 0.2, 0.3]})
    ds = RTDataset(base_df, vocab, rt_median=0.0, rt_mad=1.0)
    before = _predict(model, ds)

    # Extend with a fake new mod with parent S (inherits)
    model.extend_vocab(new_main_tokens=[("S[80]", "S", False)])
    assert "S[80]" in model.vocab.tokens
    assert model.embedding.num_embeddings == vocab.size

    # Build dataset AFTER extension (so vocab is consistent), but with the same
    # peptides — none contain S[80], so predictions should be unchanged.
    ds_after = RTDataset(base_df, vocab, rt_median=0.0, rt_mad=1.0)
    after = _predict(model, ds_after)
    assert torch.allclose(before, after, atol=1e-6)


def test_extend_fresh_keeps_base_predictions():
    torch.manual_seed(0)
    vocab = Vocabulary.default()
    model = RTModel(vocab, embedding_dim=16)

    import polars as pl
    base_df = pl.DataFrame({"seq": ["AAAR", "MCAR"], "rt": [0.1, 0.2]})
    ds = RTDataset(base_df, vocab, rt_median=0.0, rt_mad=1.0)
    before = _predict(model, ds)

    model.extend_vocab(new_main_tokens=[("Z[42]", None, True)])  # fresh, no parent
    assert "Z[42]" in model.vocab.fresh

    ds_after = RTDataset(base_df, vocab, rt_median=0.0, rt_mad=1.0)
    after = _predict(model, ds_after)
    assert torch.allclose(before, after, atol=1e-6)


def test_extend_nterm_mod_keeps_base_predictions():
    torch.manual_seed(0)
    vocab = Vocabulary.default()
    model = RTModel(vocab, embedding_dim=16)

    import polars as pl
    base_df = pl.DataFrame({"seq": ["AAAR", "MCAR"], "rt": [0.1, 0.2]})
    ds = RTDataset(base_df, vocab, rt_median=0.0, rt_mad=1.0)
    before = _predict(model, ds)

    n_before = model.nterm_mod_embedding.num_embeddings
    model.extend_vocab(new_nterm_mods=[".[42]"])
    assert model.nterm_mod_embedding.num_embeddings == n_before + 1

    after = _predict(model, ds)
    assert torch.allclose(before, after, atol=1e-6)


def test_discover_tokens_finds_unknown():
    vocab = Vocabulary.default()
    seqs = ["AAAR", "AS[80]K", "PEPTIDET[80]K", "M(UniMod:35)AR"]
    new_main, new_nterm = discover_tokens(seqs, vocab)
    assert "S[80]" in new_main and new_main["S[80]"] == "S"
    assert "T[80]" in new_main and new_main["T[80]"] == "T"
    assert new_nterm == set()


def test_discover_tokens_finds_nterm_mod():
    vocab = Vocabulary.default()
    seqs = [".[42.0]AAR", ".[308.1161]MCAR"]
    _, new_nterm = discover_tokens(seqs, vocab)
    assert ".[42]" in new_nterm
    assert ".[308]" not in new_nterm  # already in default vocab
