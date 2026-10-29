"""Library ingestion for BASE-MODEL training (internal tooling).

These loaders consume large labeled libraries (spectral-library TSVs, PSMtag
TSVs, search-engine report-library parquets), apply label hygiene, and produce
peptide-level train/val/test splits. Nothing here runs on a user's machine —
runtime adaptation goes through RTPredictor.calibrate()/fit().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from sklearn.model_selection import train_test_split


def load_spectral_library(
    path: str | Path,
    seed: int = 42,
    split_dir: str | Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """spectral-library TSV. Group by ModifiedPeptide / PeptideSequence, average
    Tr_recalibrated, split 80/10/10 at the peptide level. Output column names:
    `seq` and `rt`."""
    header = pl.read_csv(path, separator="\t", n_rows=0).columns
    mod_col = next((c for c in ("ModifiedPeptide", "ModifiedSequence") if c in header), None)
    cols = ["PeptideSequence", "Tr_recalibrated"] + ([mod_col] if mod_col else [])
    df = pl.read_csv(path, separator="\t", columns=cols)
    df = (
        df.drop_nulls("Tr_recalibrated")
          .with_columns(
              (pl.coalesce(mod_col, "PeptideSequence") if mod_col
               else pl.col("PeptideSequence")).alias("seq"))
          .drop_nulls("seq")
          .filter(pl.col("seq").str.len_chars() > 0)
          .group_by("seq")
          .agg(pl.col("Tr_recalibrated").mean().alias("rt"))
    )
    return _peptide_split(df, seed, split_dir)


def load_report_lib_parquet(
    path: str | Path,
    seed: int = 42,
    split_dir: str | Path | None = None,
    max_q_value: float = 0.01,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """search-engine report-library parquet. Drops decoys, filters to Q.Value <=
    max_q_value, canonicalizes (tag)/(UniMod:n) notation to JRIF psmtag
    format, averages RT per modified sequence. Output columns: `seq`, `rt`."""
    df = pl.read_parquet(
        path, columns=["Modified.Sequence", "RT", "Decoy", "Q.Value"],
        use_pyarrow=True,
    )
    df = (
        df.filter((pl.col("Decoy") == 0) & (pl.col("Q.Value") <= max_q_value))
          .group_by("Modified.Sequence").agg(pl.col("RT").mean().alias("rt"))
          .with_columns(
              pl.col("Modified.Sequence")
                .str.replace(r"^\(tag\)", ".[308]")
                .str.replace_all(r"\(tag\)", "[308]")
                .str.replace_all(r"\(UniMod:4\)", "[160]")
                .str.replace_all(r"\(UniMod:35\)", "[147]")
                .alias("seq"))
          .select(["seq", "rt"])
    )
    return _peptide_split(df, seed, split_dir)


def load_report_lib_parquet_im(
    path: str | Path,
    seed: int = 42,
    split_dir: str | Path | None = None,
    max_q_value: float = 0.01,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Report-library parquet for IM: same filtering and canonicalization as
    load_report_lib_parquet, IM averaged per (modified sequence, charge).
    Split is at the peptide level so no peptide's charge states straddle
    splits. Output columns: `seq`, `charge`, `im`."""
    df = pl.read_parquet(
        path, columns=["Modified.Sequence", "Precursor.Charge", "IM", "Decoy", "Q.Value"],
        use_pyarrow=True,
    )
    df = (
        df.filter((pl.col("Decoy") == 0) & (pl.col("Q.Value") <= max_q_value))
          .group_by(["Modified.Sequence", "Precursor.Charge"])
          .agg(pl.col("IM").mean().alias("im"))
          .with_columns(
              pl.col("Modified.Sequence")
                .str.replace(r"^\(tag\)", ".[308]")
                .str.replace_all(r"\(tag\)", "[308]")
                .str.replace_all(r"\(UniMod:4\)", "[160]")
                .str.replace_all(r"\(UniMod:35\)", "[147]")
                .alias("seq"),
              pl.col("Precursor.Charge").cast(pl.Int64).alias("charge"))
          .select(["seq", "charge", "im"])
    )
    return _peptide_split(df, seed, split_dir)


def load_psmtag_library(
    path: str | Path,
    seed: int = 42,
    split_dir: str | Path | None = None,
    exclude_oxm: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """FragPipe PSMtag library TSV. Output columns: `seq`, `rt`."""
    df = pl.read_csv(
        path, separator="\t",
        columns=["PeptideSequence", "ModifiedPeptideSequence", "NormalizedRetentionTime"],
    )
    df = (
        df.drop_nulls("NormalizedRetentionTime")
          .with_columns(pl.coalesce("ModifiedPeptideSequence", "PeptideSequence").alias("seq"))
          .drop_nulls("seq")
          .filter(pl.col("seq").str.len_chars() > 0)
    )
    if exclude_oxm:
        df = df.filter(~pl.col("seq").str.contains(r"M\(UniMod:35\)"))
    df = df.group_by("seq").agg(pl.col("NormalizedRetentionTime").mean().alias("rt"))
    return _peptide_split(df, seed, split_dir)


def write_peptide_splits(
    split_dir: str | Path,
    train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame,
) -> None:
    """Write `{train,val,test}_peptides.tsv` (one `seq` column, sorted) to
    `split_dir` so a split can be reproduced or audited later."""
    sp = Path(split_dir); sp.mkdir(parents=True, exist_ok=True)
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d.select(pl.col("seq").unique().sort()).write_csv(
            sp / f"{name}_peptides.tsv", separator="\t")


def _peptide_split(
    df: pl.DataFrame, seed: int, split_dir: str | Path | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Peptide-level 80/10/10 split. Pure unless `split_dir` is given, in which
    case the peptide lists are also written there (see write_peptide_splits)."""
    seqs = np.sort(df["seq"].unique().to_numpy())
    train_pep, temp_pep = train_test_split(seqs, test_size=0.2, random_state=seed)
    val_pep, test_pep = train_test_split(temp_pep, test_size=0.5, random_state=seed)

    train_pep_set = set(train_pep.tolist())
    val_pep_set = set(val_pep.tolist())
    test_pep_set = set(test_pep.tolist())
    train_df = df.filter(pl.col("seq").is_in(list(train_pep_set)))
    val_df = df.filter(pl.col("seq").is_in(list(val_pep_set)))
    test_df = df.filter(pl.col("seq").is_in(list(test_pep_set)))
    if split_dir is not None:
        write_peptide_splits(split_dir, train_df, val_df, test_df)
    return train_df, val_df, test_df


# -------- Dataset --------
