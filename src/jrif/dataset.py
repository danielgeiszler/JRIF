"""Tokenization, vocabulary, and library loaders for JRIF.

The Vocabulary object owns the token-id mapping for the main residue alphabet
plus N-term / C-term peptide-level modification mappings. It is *runtime
extensible*: when fine-tuning on a dataset that contains modification tokens
the base model has not seen, Vocabulary.extend() / extend_nterm_mod() grow the
maps and RTModel.extend_vocab grows the embedding tables to match.

Token canonicalization rules:
- Bracket-mass tokens are normalized to integer mass: 'K[308.1161]' -> 'K[308]'.
- UniMod-35 on M -> 'M[147]'; UniMod-4 on C -> 'C[160]'.
- Bare 'C' is treated as 'C[160]' in bracket and PSMtag formats (carbamido-
  methylation is fixed in those data sources). UniMod format leaves bare C
  alone (UniMod-4 must be explicit).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


MAX_LEN = 50

DEFAULT_TOKENS: dict[str, int] = {
    "<pad>": 0,
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5,
    "G": 6, "H": 7, "I": 8, "K": 9, "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
    "M[147]": 21,
    "C[160]": 22,
    "K[308]": 23,
}

DEFAULT_NTERM_MOD_TOKENS: dict[str, int] = {
    "": 0,
    ".[308]": 1,
}

DEFAULT_CTERM_MOD_TOKENS: dict[str, int] = {"": 0}

DEFAULT_PARENT_OF: dict[str, str] = {
    "M[147]": "M",
    "C[160]": "C",
    "K[308]": "K",
}

# K[308]'s side-chain tag changes chemistry too much to inherit K's physico
# features. Only the learned embedding picks up signal for it.
DEFAULT_FRESH: set[str] = {"K[308]"}

CANONICAL_AAS: tuple[str, ...] = (
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
)

_MOD_PATTERN = re.compile(r"([A-Z])\[[\d.]+\]|([A-Z])")
_UNIMOD_PATTERN = re.compile(r"([A-Z])\(UniMod:\d+\)|([A-Z])")
_PSMTAG_NTERM = re.compile(r"^\.\[[\d.]+\]")
_PSMTAG_PATTERN = re.compile(
    r"([A-Z])\[[\d.]+\]|([A-Z])\(UniMod:\d+\)|([A-Z])"
)


@dataclass
class Vocabulary:
    """Token-id maps + lineage tracking. Mutate via extend* methods."""
    tokens: dict[str, int]
    nterm_mod_tokens: dict[str, int]
    cterm_mod_tokens: dict[str, int]
    parent_of: dict[str, str] = field(default_factory=dict)
    fresh: set[str] = field(default_factory=set)

    @classmethod
    def default(cls) -> "Vocabulary":
        return cls(
            tokens=dict(DEFAULT_TOKENS),
            nterm_mod_tokens=dict(DEFAULT_NTERM_MOD_TOKENS),
            cterm_mod_tokens=dict(DEFAULT_CTERM_MOD_TOKENS),
            parent_of=dict(DEFAULT_PARENT_OF),
            fresh=set(DEFAULT_FRESH),
        )

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def nterm_size(self) -> int:
        return len(self.nterm_mod_tokens)

    @property
    def cterm_size(self) -> int:
        return len(self.cterm_mod_tokens)

    def has(self, token: str) -> bool:
        return token in self.tokens

    def extend(self, token: str, parent_aa: Optional[str], fresh: bool = False) -> int:
        if token in self.tokens:
            return self.tokens[token]
        new_id = len(self.tokens)
        self.tokens[token] = new_id
        if parent_aa is not None and not fresh:
            self.parent_of[token] = parent_aa
        if fresh:
            self.fresh.add(token)
        return new_id

    def extend_nterm_mod(self, mod: str) -> int:
        if mod in self.nterm_mod_tokens:
            return self.nterm_mod_tokens[mod]
        new_id = len(self.nterm_mod_tokens)
        self.nterm_mod_tokens[mod] = new_id
        return new_id

    def extend_cterm_mod(self, mod: str) -> int:
        if mod in self.cterm_mod_tokens:
            return self.cterm_mod_tokens[mod]
        new_id = len(self.cterm_mod_tokens)
        self.cterm_mod_tokens[mod] = new_id
        return new_id

    def parent_id(self, token: str) -> Optional[int]:
        parent = self.parent_of.get(token)
        if parent is None:
            return None
        return self.tokens.get(parent)

    def to_dict(self) -> dict:
        return {
            "tokens": dict(self.tokens),
            "nterm_mod_tokens": dict(self.nterm_mod_tokens),
            "cterm_mod_tokens": dict(self.cterm_mod_tokens),
            "parent_of": dict(self.parent_of),
            "fresh": sorted(self.fresh),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        return cls(
            tokens=dict(d["tokens"]),
            nterm_mod_tokens=dict(d["nterm_mod_tokens"]),
            cterm_mod_tokens=dict(d["cterm_mod_tokens"]),
            parent_of=dict(d.get("parent_of", {})),
            fresh=set(d.get("fresh", [])),
        )


# -------- Tokenization --------

def _detect_format(seq: str) -> str:
    if seq.startswith(".["):
        return "psmtag"
    if "UniMod:" in seq:
        return "unimod"
    return "bracket"


def _iter_main_matches(body: str, fmt: str) -> Iterable[re.Match]:
    if fmt == "psmtag":
        return _PSMTAG_PATTERN.finditer(body)
    if fmt == "unimod":
        return _UNIMOD_PATTERN.finditer(body)
    return _MOD_PATTERN.finditer(body)


def _strip_nterm(seq: str) -> tuple[str, Optional[str]]:
    m = _PSMTAG_NTERM.match(seq)
    if not m:
        return seq, None
    return seq[m.end():], _canon_nterm(m.group(0)[2:-1])


def _canon_bracket(aa: str, content: str) -> str:
    try:
        return f"{aa}[{int(float(content))}]"
    except ValueError:
        return f"{aa}[{content}]"


def _canon_nterm(content: str) -> str:
    try:
        return f".[{int(float(content))}]"
    except ValueError:
        return f".[{content}]"


def _resolve_main(aa: Optional[str], full: str, fmt: str) -> Optional[str]:
    if not aa:
        return None
    if "(UniMod:35)" in full and aa == "M":
        return "M[147]"
    if "(UniMod:4)" in full and aa == "C":
        return "C[160]"
    if "[" in full:
        i = full.index("["); j = full.index("]", i)
        return _canon_bracket(aa, full[i+1:j])
    if aa == "C" and fmt != "unimod":
        return "C[160]"
    return aa


def _pad(tokens: list[int]) -> list[int]:
    if len(tokens) >= MAX_LEN:
        return tokens[:MAX_LEN]
    return tokens + [0] * (MAX_LEN - len(tokens))


def tokenize(seq: str, vocab: Vocabulary) -> tuple[list[int], int, int]:
    """Tokenize one peptide sequence with the given Vocabulary.

    Returns (token_ids[MAX_LEN], nterm_mod_id, cterm_mod_id). Tokens not in
    `vocab` fall back to the bare AA id when possible, otherwise are dropped.
    """
    if not isinstance(seq, str) or not seq:
        return _pad([]), 0, 0
    body, nterm_mod = _strip_nterm(seq)
    nterm_id = vocab.nterm_mod_tokens.get(nterm_mod, 0) if nterm_mod else 0
    fmt = _detect_format(seq)
    out: list[int] = []
    for m in _iter_main_matches(body, fmt):
        aa = next((g for g in m.groups() if g), None)
        token = _resolve_main(aa, m.group(0), fmt)
        if token is None:
            continue
        idx = vocab.tokens.get(token)
        if idx is None and aa in vocab.tokens:
            idx = vocab.tokens[aa]
        if idx is None:
            continue
        out.append(idx)
    return _pad(out), nterm_id, 0


def discover_tokens(
    seqs: Iterable[str], vocab: Vocabulary,
) -> tuple[dict[str, Optional[str]], set[str]]:
    """Walk sequences and return tokens not yet present in `vocab`.

    Returns:
      (new_main_tokens, new_nterm_mods)
      where new_main_tokens maps each unknown token to its inferred parent AA
      (the bare residue letter for X[NNN]-form tokens; None otherwise).
    """
    new_main: dict[str, Optional[str]] = {}
    new_nterm: set[str] = set()
    for seq in seqs:
        if not isinstance(seq, str) or not seq:
            continue
        body, nterm_mod = _strip_nterm(seq)
        if nterm_mod and nterm_mod not in vocab.nterm_mod_tokens:
            new_nterm.add(nterm_mod)
        fmt = _detect_format(seq)
        for m in _iter_main_matches(body, fmt):
            aa = next((g for g in m.groups() if g), None)
            token = _resolve_main(aa, m.group(0), fmt)
            if token is None or token in vocab.tokens or token in new_main:
                continue
            parent = aa if aa and aa in vocab.tokens else None
            new_main[token] = parent
    return new_main, new_nterm


# -------- DataFrame helpers --------

def to_polars(df) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)!r}")


def fit_rt_scale(rts: np.ndarray) -> tuple[float, float]:
    """Robust z-score scale: median + MAD. MAD of 0 collapses to 1.0."""
    med = float(np.median(rts))
    mad = float(np.median(np.abs(rts - med)))
    if mad == 0:
        mad = 1.0
    return med, mad


# -------- Library loaders --------

def load_spectral_library(
    path: str | Path,
    seed: int = 42,
    split_dir: str | Path = "splits",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """DIA-NN library TSV. Group by ModifiedPeptide / PeptideSequence, average
    Tr_recalibrated, split 80/10/10 at the peptide level. Output column names:
    `seq` and `rt`."""
    df = pl.read_csv(
        path, separator="\t",
        columns=["PeptideSequence", "ModifiedPeptide", "Tr_recalibrated"],
    )
    df = (
        df.drop_nulls("Tr_recalibrated")
          .with_columns(pl.coalesce("ModifiedPeptide", "PeptideSequence").alias("seq"))
          .drop_nulls("seq")
          .filter(pl.col("seq").str.len_chars() > 0)
          .group_by("seq")
          .agg(pl.col("Tr_recalibrated").mean().alias("rt"))
    )
    return _peptide_split(df, seed, split_dir)


def load_psmtag_library(
    path: str | Path,
    seed: int = 42,
    split_dir: str | Path = "splits_psmtag",
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


def _peptide_split(
    df: pl.DataFrame, seed: int, split_dir: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    seqs = np.sort(df["seq"].unique().to_numpy())
    train_pep, temp_pep = train_test_split(seqs, test_size=0.2, random_state=seed)
    val_pep, test_pep = train_test_split(temp_pep, test_size=0.5, random_state=seed)

    sp = Path(split_dir); sp.mkdir(parents=True, exist_ok=True)
    for name, peps in [("train", train_pep), ("val", val_pep), ("test", test_pep)]:
        pl.DataFrame({"seq": peps}).write_csv(sp / f"{name}_peptides.tsv", separator="\t")

    train_pep_set = set(train_pep.tolist())
    val_pep_set = set(val_pep.tolist())
    test_pep_set = set(test_pep.tolist())
    train_df = df.filter(pl.col("seq").is_in(list(train_pep_set)))
    val_df = df.filter(pl.col("seq").is_in(list(val_pep_set)))
    test_df = df.filter(pl.col("seq").is_in(list(test_pep_set)))
    return train_df, val_df, test_df


# -------- Dataset --------

class RTDataset(Dataset):
    """Tokenized peptides + z-scored RT targets + terminal mod ids.

    __getitem__ returns (tokens, nterm_mod_id, cterm_mod_id, target). The
    target is z-scored against (rt_median, rt_mad); pass the TRAIN scale when
    building val/test/predict datasets.

    Pass `rt_col=None` for predict-only datasets where RTs are placeholders.
    """

    def __init__(
        self,
        df,
        vocab: Vocabulary,
        *,
        seq_col: str = "seq",
        rt_col: Optional[str] = "rt",
        rt_median: Optional[float] = None,
        rt_mad: Optional[float] = None,
    ):
        df = to_polars(df)
        seqs = df[seq_col].to_list()
        n = len(seqs)

        tok_arr = np.zeros((n, MAX_LEN), dtype=np.int64)
        nterm_arr = np.zeros(n, dtype=np.int64)
        cterm_arr = np.zeros(n, dtype=np.int64)
        for i, s in enumerate(seqs):
            t, ni, ci = tokenize(s, vocab)
            tok_arr[i, :] = t
            nterm_arr[i] = ni
            cterm_arr[i] = ci

        self.sequences = torch.from_numpy(tok_arr)
        self.nterm_mod_ids = torch.from_numpy(nterm_arr)
        self.cterm_mod_ids = torch.from_numpy(cterm_arr)

        if rt_col is None:
            self.rt_median = float(rt_median if rt_median is not None else 0.0)
            self.rt_mad = float(rt_mad if rt_mad is not None else 1.0)
            self.targets = torch.zeros(n, dtype=torch.float32)
        else:
            rts = df[rt_col].to_numpy().astype("float32")
            if rt_median is None or rt_mad is None:
                rt_median, rt_mad = fit_rt_scale(rts)
            self.rt_median = float(rt_median)
            self.rt_mad = float(rt_mad)
            self.targets = torch.from_numpy((rts - self.rt_median) / self.rt_mad)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return (
            self.sequences[idx],
            self.nterm_mod_ids[idx],
            self.cterm_mod_ids[idx],
            self.targets[idx],
        )
