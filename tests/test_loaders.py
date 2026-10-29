"""Training loaders: the peptide split is pure unless split_dir is given."""

import polars as pl

from jrif.training.loaders import _peptide_split, load_psmtag_library


def _toy_df(n: int = 200) -> pl.DataFrame:
    return pl.DataFrame({
        "seq": [f"PEPTIDE{i:04d}K" for i in range(n)],
        "rt": [float(i) for i in range(n)],
    })


def test_peptide_split_writes_nothing_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train, val, test = _peptide_split(_toy_df(), seed=0, split_dir=None)
    assert list(tmp_path.iterdir()) == []
    assert len(train) + len(val) + len(test) == 200
    s_tr, s_va, s_te = (set(d["seq"]) for d in (train, val, test))
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)


def test_peptide_split_writes_when_asked(tmp_path):
    split_dir = tmp_path / "splits"
    train, val, test = _peptide_split(_toy_df(), seed=0, split_dir=split_dir)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        written = pl.read_csv(split_dir / f"{name}_peptides.tsv", separator="\t")
        assert set(written["seq"]) == set(d["seq"])


def test_loader_is_side_effect_free(tmp_path, monkeypatch):
    lib = tmp_path / "lib.tsv"
    df = _toy_df()
    pl.DataFrame({
        "PeptideSequence": df["seq"],
        "ModifiedPeptideSequence": df["seq"],
        "NormalizedRetentionTime": df["rt"],
    }).write_csv(lib, separator="\t")
    monkeypatch.chdir(tmp_path)
    load_psmtag_library(lib, seed=0)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["lib.tsv"]
