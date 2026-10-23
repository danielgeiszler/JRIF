"""Predict RT for every unique peptide in a spectral library TSV and write
back the same TSV with the RT column replaced by predictions.

Auto-detects PSMtag (ModifiedPeptideSequence / NormalizedRetentionTime) vs
spectral-library (ModifiedPeptide / Tr_recalibrated) format. Use --add-psmtag if your
library is implicitly PSMtag-tagged but not annotated in the sequence string.
"""

import argparse

import polars as pl

from jrif.api import RTPredictor
from jrif.io import FORMATS, apply_psmtag, detect_format, read_library_tsv, write_library_tsv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-tsv", required=True)
    p.add_argument("--output-tsv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--add-psmtag", action="store_true",
                   help="Annotate sequences with PSMtag before tokenization "
                        "(for libraries where PSMtag is implicit but not "
                        "written in the sequence strings).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    print(f"Loading library: {args.input_tsv}")
    df = read_library_tsv(args.input_tsv)
    fmt = detect_format(df.columns)
    peptide_col, rt_col = FORMATS[fmt]
    print(f"  format: {fmt}  peptide_col={peptide_col}  rt_col={rt_col}")
    print(f"  {len(df):,} rows, {df[peptide_col].n_unique():,} unique peptides")

    unique_peps = df.select(peptide_col).unique().with_row_index("__idx__")
    if args.add_psmtag:
        unique_peps = unique_peps.with_columns(
            pl.col(peptide_col).map_elements(apply_psmtag, return_dtype=pl.Utf8).alias("seq")
        )
        print("  (added PSMtag annotations to sequences for prediction)")
    else:
        unique_peps = unique_peps.with_columns(pl.col(peptide_col).alias("seq"))

    print(f"Loading checkpoint: {args.checkpoint}")
    predictor = RTPredictor.from_checkpoint(args.checkpoint, device=args.device)
    print(f"  RT scale: median={predictor.rt_median:.3f}, MAD={predictor.rt_mad:.3f}")

    preds = predictor.predict(
        unique_peps.select("seq"),
        seq_col="seq",
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Predicted {len(preds):,} peptides. RT range: [{preds.min():.3f}, {preds.max():.3f}]")

    pep_to_rt = dict(zip(unique_peps[peptide_col].to_list(), preds.tolist()))
    out = df.with_columns(
        pl.col(peptide_col).map_elements(
            lambda s: pep_to_rt.get(s), return_dtype=pl.Float64,
        ).alias(rt_col)
    )

    missing = out[rt_col].is_null().sum()
    if missing:
        print(f"WARN: {missing} rows have no prediction (NaN peptide?)")

    write_library_tsv(out, args.output_tsv)
    print(f"Wrote {len(out):,} rows to {args.output_tsv}")


if __name__ == "__main__":
    main()
