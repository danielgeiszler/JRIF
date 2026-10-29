"""predict() dedupes repeated peptides and chunks tokenization.

Neither may change the numbers: every chunk size / batch size must reproduce
a plain row-by-row forward pass, in input order.
"""

import sys

import numpy as np
import polars as pl
import pytest
import torch

import jrif.api
import jrif.im
from jrif import TokenizationError
from jrif.api import RTPredictor
from jrif.dataset import tokenize, unique_rows
from jrif.im import IMPredictor


def _library(n_unique=40, repeats=4, seed=0):
    """Spectral-library shape: each peptide appears several times, shuffled."""
    rng = np.random.default_rng(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    peps = ["".join(rng.choice(aas, int(rng.integers(6, 16)))) for _ in range(n_unique)]
    seqs = [p for p in peps for _ in range(repeats)]
    rng.shuffle(seqs)
    charges = rng.choice([2, 3], len(seqs))
    return pl.DataFrame({"seq": seqs, "charge": charges})


def _rt_reference(p, df):
    """One forward pass per row, no dedupe, no chunking."""
    p.model.eval()
    with torch.inference_mode():
        z = np.array([p.model(torch.tensor([tokenize(s, p.vocab)])).item()
                      for s in df["seq"].to_list()])
    out = z * p.rt_mad + p.rt_median
    return np.interp(out, *p.rt_warp) if p.rt_warp is not None else out


def _im_reference(p, df):
    p.model.eval()
    with torch.inference_mode():
        z = np.array([p.model(torch.tensor([tokenize(s, p.vocab)]),
                              torch.tensor([float(c) - 2.0])).item()
                      for s, c in zip(df["seq"].to_list(), df["charge"].to_list())])
    return z * p.im_mad + p.im_median


def test_unique_rows_inverse_reconstructs():
    df = pl.DataFrame({"seq": ["B", "A", "B", "A", "C"], "charge": [2, 2, 2, 3, 2]})
    uniq, inv = unique_rows(df, ["seq"])
    assert uniq["seq"].to_list() == ["B", "A", "C"]  # first-seen order
    assert uniq["seq"].gather(inv).to_list() == df["seq"].to_list()
    uniq2, inv2 = unique_rows(df, ["seq", "charge"])
    assert uniq2.height == 4
    assert uniq2[inv2.tolist()].equals(df)


@pytest.mark.parametrize("chunk,batch", [(7, 3), (1, 1), (1000, 2048)])
def test_rt_dedupe_and_chunking_match_reference(monkeypatch, chunk, batch):
    torch.manual_seed(0)
    df = _library()
    p = RTPredictor.from_blank()
    p.rt_median, p.rt_mad = 30.0, 7.0
    ref = _rt_reference(p, df)
    monkeypatch.setattr(jrif.api, "PREDICT_CHUNK_SIZE", chunk)
    got = p.predict(df, device="cpu", batch_size=batch)
    assert got.shape == (df.height,)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)


def test_rt_warp_applies_per_row(monkeypatch):
    torch.manual_seed(0)
    df = _library()
    p = RTPredictor.from_blank()
    raw = _rt_reference(p, df)
    p.rt_warp = (np.array([raw.min() - 1, raw.max() + 1]), np.array([0.0, 100.0]))
    monkeypatch.setattr(jrif.api, "PREDICT_CHUNK_SIZE", 5)
    np.testing.assert_allclose(p.predict(df, device="cpu"), _rt_reference(p, df), atol=1e-5)


def test_rt_predict_empty_and_pandas():
    p = RTPredictor.from_blank()
    assert p.predict(pl.DataFrame({"seq": []}, schema={"seq": pl.Utf8}),
                     device="cpu").shape == (0,)
    df = _library(n_unique=5, repeats=2)
    np.testing.assert_allclose(p.predict(df.to_pandas(), device="cpu"),
                               p.predict(df, device="cpu"), atol=1e-6)


def test_predict_raises_on_unrepresentable_row_in_any_chunk(monkeypatch):
    p = RTPredictor.from_blank()
    df = pl.DataFrame({"seq": [f"PEPTIDE{a}K" for a in "ACDEFGH"] + ["PES[80]K"]})
    monkeypatch.setattr(jrif.api, "PREDICT_CHUNK_SIZE", 3)
    with pytest.raises(TokenizationError, match=r"S\[80\]"):
        p.predict(df, device="cpu")


def test_im_dedupes_on_seq_and_charge(monkeypatch):
    torch.manual_seed(0)
    df = _library()
    p = IMPredictor.from_blank()
    ref = _im_reference(p, df)
    monkeypatch.setattr(jrif.im, "PREDICT_CHUNK_SIZE", 7)
    np.testing.assert_allclose(p.predict(df, device="cpu", batch_size=3), ref, rtol=0, atol=1e-5)
    # Same peptide at a different charge must not collapse to one prediction.
    seq = df["seq"][0]
    a, b = p.predict(pl.DataFrame({"seq": [seq, seq], "charge": [2, 3]}), device="cpu")
    assert a != pytest.approx(b)


def test_predict_library_cli_roundtrip(tmp_path, monkeypatch):
    from jrif import predict_library

    torch.manual_seed(0)
    p = RTPredictor.from_blank()
    ckpt = tmp_path / "m.pt"
    p.save(ckpt)
    lib = _library(n_unique=10, repeats=3)
    lib = pl.DataFrame({
        "ModifiedPeptide": lib["seq"].to_list() + [None],
        "PrecursorCharge": lib["charge"].to_list() + [2],
        "Tr_recalibrated": [0.0] * (lib.height + 1),
    })
    src, dst = tmp_path / "in.tsv", tmp_path / "out.tsv"
    lib.write_csv(src, separator="\t")
    monkeypatch.setattr(sys, "argv", [
        "predict_library", "--input-tsv", str(src), "--output-tsv", str(dst),
        "--checkpoint", str(ckpt), "--device", "cpu"])
    predict_library.main()

    out = pl.read_csv(dst, separator="\t")
    assert out.height == lib.height
    assert out["Tr_recalibrated"][-1] is None
    expect = _rt_reference(p, lib.head(-1).select(pl.col("ModifiedPeptide").alias("seq")))
    np.testing.assert_allclose(out["Tr_recalibrated"].head(-1).to_numpy(), expect, atol=1e-4)
