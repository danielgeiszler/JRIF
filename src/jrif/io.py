"""Library-TSV I/O helpers: format detection (PSMtag vs spectral-library) and
PSMtag annotation for libraries where the tag is implicit."""

from __future__ import annotations

from pathlib import Path

import polars as pl


# Per-format (peptide_col, rt_col) on the source TSV.
FORMATS: dict[str, tuple[str, str]] = {
    "psmtag": ("ModifiedPeptideSequence", "NormalizedRetentionTime"),
    "speclib": ("ModifiedPeptide", "Tr_recalibrated"),
}


def detect_format(columns) -> str:
    cols = set(columns)
    if {"ModifiedPeptideSequence", "NormalizedRetentionTime"}.issubset(cols):
        return "psmtag"
    if {"ModifiedPeptide", "Tr_recalibrated"}.issubset(cols):
        return "speclib"
    raise SystemExit(
        f"Unrecognized library format. Need either "
        f"[ModifiedPeptideSequence, NormalizedRetentionTime] (PSMtag) or "
        f"[ModifiedPeptide, Tr_recalibrated] (spectral library). Got columns: {sorted(cols)}"
    )


def apply_psmtag(seq: str) -> str:
    """Annotate a plain modified-peptide string with PSMtag: prepend
    `.[308.1161]` and tag every K not already bracket-modified."""
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


def read_library_tsv(path: str | Path) -> pl.DataFrame:
    """Read a FragPipe / spectral-library TSV with full column set preserved."""
    return pl.read_csv(path, separator="\t", infer_schema_length=1000)


def write_library_tsv(df: pl.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out, separator="\t")
