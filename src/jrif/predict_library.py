"""Predict RT for every unique peptide in a spectral library TSV and write
back the same TSV with the RT column replaced by predictions.

Auto-detects PSMtag (ModifiedPeptideSequence / NormalizedRetentionTime) vs
spectral-library (ModifiedPeptide / Tr_recalibrated) format. Use --add-psmtag if your
library is implicitly PSMtag-tagged but not annotated in the sequence string.
"""

import argparse

import numpy as np
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

    seqs = df[peptide_col]
    if args.add_psmtag:
        uniq = seqs.drop_nulls().unique().to_list()
        seqs = seqs.replace_strict({s: apply_psmtag(s) for s in uniq},
                                   default=None, return_dtype=pl.Utf8)
        print("  (added PSMtag annotations to sequences for prediction)")

    print(f"Loading checkpoint: {args.checkpoint}")
    predictor = RTPredictor.from_checkpoint(args.checkpoint, device=args.device)
    print(f"  RT scale: median={predictor.rt_median:.3f}, MAD={predictor.rt_mad:.3f}")

    # predict() dedupes peptides internally and broadcasts back to rows.
    # Rows with no peptide get no prediction; any other row predict() cannot
    # represent raises.
    has_seq = seqs.is_not_null().to_numpy()
    rt = np.full(len(df), np.nan)
    rt[has_seq] = predictor.predict(
        pl.DataFrame({"seq": seqs.filter(pl.Series(has_seq))}),
        seq_col="seq",
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Predicted {int(has_seq.sum()):,} rows. RT range: "
          f"[{np.nanmin(rt):.3f}, {np.nanmax(rt):.3f}]")
    out = df.with_columns(pl.Series(rt_col, rt).fill_nan(None))

    missing = int((~has_seq).sum())
    if missing:
        print(f"WARN: {missing} rows have no peptide and no prediction")

    write_library_tsv(out, args.output_tsv)
    print(f"Wrote {len(out):,} rows to {args.output_tsv}")


if __name__ == "__main__":
    main()
