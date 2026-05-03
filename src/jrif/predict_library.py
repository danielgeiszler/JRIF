"""Predict RT for every unique peptide in a spectral library TSV, replace the RT column.

- Input: FragPipe/DIA-NN-style library TSV. Auto-detects PSMtag format
  (ModifiedPeptideSequence / NormalizedRetentionTime) vs DIA-NN format
  (ModifiedPeptide / Tr_recalibrated).
- Output: same columns, but the RT column has our predicted values (back-transformed
  from z-scored model output using the checkpoint's rt_median / rt_mad).
- --add-psmtag: useful when the caller has a library where PSMtag is *implicit*
  but not annotated in the peptide string. Adds `.[308.1161]` to every N-terminus
  and `[308.1161]` to every K before tokenization.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from jrif.dataset import RTDataset
from jrif.model import RTModel


# Columns used by each format: (peptide_col, rt_col).
FORMATS = {
    "psmtag": ("ModifiedPeptideSequence", "NormalizedRetentionTime"),
    "diann": ("ModifiedPeptide", "Tr_recalibrated"),
}


def detect_format(columns) -> str:
    cols = set(columns)
    if {"ModifiedPeptideSequence", "NormalizedRetentionTime"}.issubset(cols):
        return "psmtag"
    if {"ModifiedPeptide", "Tr_recalibrated"}.issubset(cols):
        return "diann"
    raise SystemExit(
        f"Unrecognized library format. Need either "
        f"[ModifiedPeptideSequence, NormalizedRetentionTime] (PSMtag) or "
        f"[ModifiedPeptide, Tr_recalibrated] (DIA-NN). Got columns: {sorted(cols)}"
    )


def apply_psmtag(seq: str) -> str:
    """Annotate a plain modified-peptide string with PSMtag: prepend .[308.1161]
    and tag every K that isn't already bracket-modified."""
    if not isinstance(seq, str) or not seq:
        return seq
    tagged = seq if seq.startswith(".[") else ".[308.1161]" + seq
    out: list[str] = []
    i = 0
    while i < len(tagged):
        out.append(tagged[i])
        if tagged[i] == "K" and not (i + 1 < len(tagged) and tagged[i + 1] == "["):
            out.append("[308.1161]")
        i += 1
    return "".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-tsv", required=True)
    p.add_argument("--output-tsv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--add-psmtag", action="store_true",
                   help="Annotate sequences with PSMtag before tokenization (for libraries "
                        "where PSMtag is implicit but not written in the sequence strings).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_float32_matmul_precision("high")

    print(f"Loading library: {args.input_tsv}")
    df = pd.read_csv(args.input_tsv, sep="\t")
    fmt = detect_format(df.columns)
    peptide_col, rt_col = FORMATS[fmt]
    print(f"  format: {fmt}  peptide_col={peptide_col}  rt_col={rt_col}")
    print(f"  {len(df):,} rows, {df[peptide_col].nunique():,} unique peptides")

    # Deduplicate peptides for prediction
    unique_peps = df[[peptide_col]].drop_duplicates().reset_index(drop=True)
    if args.add_psmtag:
        unique_peps["seq"] = unique_peps[peptide_col].astype(str).apply(apply_psmtag)
        print("  (added PSMtag annotations to sequences for prediction)")
    else:
        unique_peps["seq"] = unique_peps[peptide_col].astype(str)
    unique_peps["Retention"] = 0.0  # placeholder (RTDataset requires column; we don't use the target)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=True)
    rt_med = ckpt.get("rt_median"); rt_mad = ckpt.get("rt_mad")
    if rt_med is None or rt_mad is None:
        raise SystemExit("Checkpoint is missing rt_median/rt_mad — retrain with current train_* scripts.")
    print(f"  RT scale: median={rt_med:.3f}, MAD={rt_mad:.3f}")

    model = RTModel().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = RTDataset(unique_peps, rt_median=rt_med, rt_mad=rt_mad)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for seqs, nterm, cterm, _ in loader:
            z = model(seqs.to(device), nterm.to(device), cterm.to(device)).cpu()
            preds.append(z * rt_mad + rt_med)        # back-transform to iRT / seconds
    preds_arr = torch.cat(preds).numpy()
    print(f"Predicted {len(preds_arr):,} peptides. RT range: [{preds_arr.min():.3f}, {preds_arr.max():.3f}]")

    unique_peps["_pred_rt"] = preds_arr
    pep_to_rt = dict(zip(unique_peps[peptide_col], unique_peps["_pred_rt"]))
    df[rt_col] = df[peptide_col].map(pep_to_rt)

    missing = df[rt_col].isna().sum()
    if missing:
        print(f"WARN: {missing} rows have no prediction (NaN peptide?)")

    out_path = Path(args.output_tsv); out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
