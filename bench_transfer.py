"""Does pretraining on the other dataset help? Four arms, unified dilated_cnn:

  scratch_psmtag : train from scratch on PSMtag, eval PSMtag test
  lf_to_psmtag   : pretrain LF (lr 1e-3) -> fine-tune PSMtag (lr 1e-4), eval PSMtag test
  scratch_lf     : train from scratch on LF, eval LF test
  psmtag_to_lf   : pretrain PSMtag -> fine-tune LF, eval LF test

Data:
  LF     = JD_LF.tsv (89k unmodified peptides, iRT-like scale, no decoys).
  PSMtag = 20260902_apex/report-lib.parquet filtered to Decoy==0 AND
           Q.Value<=0.01 (the unfiltered library is 6.4% decoys — their RTs
           are not empirical).

All arms: shipping recipe (mixture loss, val-MAE selection, Muon+AdamW,
cosine+warmup, patience 20), batch_size="auto", peptide-level 80/10/10 split
(seed 42) per dataset. Metrics in each dataset's native RT units.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from jrif.api import RTPredictor

LF_DATA = "/Users/danielgeiszler/Documents/jmod_tests/20260120_entrapments/libs/JD_LF.tsv"
PSM_DATA = "/Users/danielgeiszler/Documents/jmod_tests/20260902_apex/report-lib.parquet"

_MOD_MAP = {"(UniMod:4)": "[160]", "(UniMod:35)": "[147]", "(tag)": "[308]"}


def convert_seq(s: str) -> str:
    if s.startswith("(tag)"):
        s = ".[308]" + s[len("(tag)"):]
    for k, v in _MOD_MAP.items():
        s = s.replace(k, v)
    return s


def load_lf() -> pl.DataFrame:
    return (
        pl.scan_csv(LF_DATA, separator="\t")
          .select(["PeptideSequence", "Tr_recalibrated"])
          .collect()
          .group_by("PeptideSequence")
          .agg(pl.col("Tr_recalibrated").mean().alias("rt"))
          .rename({"PeptideSequence": "seq"})
          .sort("seq")
    )


def load_psmtag() -> pl.DataFrame:
    df = pl.from_arrow(pq.read_table(
        PSM_DATA, columns=["Modified.Sequence", "RT", "Decoy", "Q.Value"]))
    return (
        df.filter((pl.col("Decoy") == 0) & (pl.col("Q.Value") <= 0.01))
          .group_by("Modified.Sequence").agg(pl.col("RT").mean().alias("rt"))
          .with_columns(pl.col("Modified.Sequence")
                        .map_elements(convert_seq, return_dtype=pl.Utf8).alias("seq"))
          .select(["seq", "rt"])
          .sort("seq")
    )


def split(df: pl.DataFrame, seed: int = 42):
    seqs = df["seq"].to_list()
    rng = random.Random(seed)
    rng.shuffle(seqs)
    n = len(seqs)
    test_s = set(seqs[:int(n * 0.1)])
    val_s = set(seqs[int(n * 0.1):int(n * 0.2)])
    return (
        df.filter(~pl.col("seq").is_in(list(test_s | val_s))),
        df.filter(pl.col("seq").is_in(list(val_s))),
        df.filter(pl.col("seq").is_in(list(test_s))),
    )


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    err = np.abs(p - y)
    return {"r2": float(r2_score(y, p)), "spearman": float(spearmanr(y, p).statistic),
            "mae": float(err.mean()), "d95": float(np.quantile(err, 0.95))}


def fit_eval(pred: RTPredictor, train, val, test, *, lr, seed, device) -> dict:
    pred.fit(train, val_df=val, epochs=150, patience=20, lr=lr,
             batch_size="auto", loss="mixture", selection_metric="mae",
             seed=seed, device=device, verbose=False)
    out = metrics(test["rt"].to_numpy(), pred.predict(test, device=device))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results/bench_transfer.json")
    args = ap.parse_args()

    lf = load_lf()
    psm = load_psmtag()
    lf_tr, lf_va, lf_te = split(lf)
    psm_tr, psm_va, psm_te = split(psm)
    print(f"LF  train/val/test: {lf_tr.height}/{lf_va.height}/{lf_te.height}", flush=True)
    print(f"PSM train/val/test: {psm_tr.height}/{psm_va.height}/{psm_te.height}", flush=True)

    results = {"lf_n": lf.height, "psm_n": psm.height, "arms": {}}

    for seed in args.seeds:
        print(f"=== seed {seed}", flush=True)

        print("  scratch_psmtag", flush=True)
        p = RTPredictor.from_blank(arch="dilated_cnn", unified_termini=True)
        r = fit_eval(p, psm_tr, psm_va, psm_te, lr=1e-3, seed=seed, device=args.device)
        results["arms"].setdefault("scratch_psmtag", []).append(r)
        print(f"    {r}", flush=True)

        print("  lf_to_psmtag", flush=True)
        p = RTPredictor.from_blank(arch="dilated_cnn", unified_termini=True)
        p.fit(lf_tr, val_df=lf_va, epochs=150, patience=20, lr=1e-3,
              batch_size="auto", loss="mixture", selection_metric="mae",
              seed=seed, device=args.device, verbose=False)
        lf_pretrain_test = metrics(lf_te["rt"].to_numpy(), p.predict(lf_te, device=args.device))
        r = fit_eval(p, psm_tr, psm_va, psm_te, lr=1e-4, seed=seed, device=args.device)
        r["pretrain_lf_test"] = lf_pretrain_test
        results["arms"].setdefault("lf_to_psmtag", []).append(r)
        print(f"    {r}", flush=True)

        print("  scratch_lf", flush=True)
        results["arms"].setdefault("scratch_lf", []).append(lf_pretrain_test)
        print(f"    (reused pretrain eval) {lf_pretrain_test}", flush=True)

        print("  psmtag_to_lf", flush=True)
        p = RTPredictor.from_blank(arch="dilated_cnn", unified_termini=True)
        p.fit(psm_tr, val_df=psm_va, epochs=150, patience=20, lr=1e-3,
              batch_size="auto", loss="mixture", selection_metric="mae",
              seed=seed, device=args.device, verbose=False)
        r = fit_eval(p, lf_tr, lf_va, lf_te, lr=1e-4, seed=seed, device=args.device)
        results["arms"].setdefault("psmtag_to_lf", []).append(r)
        print(f"    {r}", flush=True)

        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"  (saved through seed {seed})", flush=True)

    for name, runs in results["arms"].items():
        maes = [r["mae"] for r in runs]
        print(f"{name:16s} test MAE {np.mean(maes):.5f} ± {np.std(maes):.5f} (n={len(maes)})")


if __name__ == "__main__":
    main()
