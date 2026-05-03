"""Tokenization, vocabulary, and library loaders for JRIF."""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


MAX_LEN = 50

VOCAB = {
    "<pad>": 0,
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5,
    "G": 6, "H": 7, "I": 8, "K": 9, "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
    "M[147]": 21,   # oxidized methionine
    "C[160]": 22,   # carbamidomethyl cysteine
    "K[308]": 23,   # PSMtag-tagged lysine (side-chain 308.1161 mod)
}
VOCAB_SIZE = len(VOCAB)
K_IDX = VOCAB["K"]
K_TAG_IDX = VOCAB["K[308]"]

# N-term alpha-amine mod vocab (0 = no mod).
NTERM_MOD_VOCAB = {
    "": 0,
    ".[308.1161]": 1,   # PSMtag N-terminus
}
NTERM_MOD_VOCAB_SIZE = 8  # room for future mods

# C-term alpha-carboxyl mod vocab. No data yet; included for future extension.
CTERM_MOD_VOCAB = {"": 0}
CTERM_MOD_VOCAB_SIZE = 8


_MOD_PATTERN = re.compile(r"([A-Z])\[[\d.]+\]|([A-Z])")
_UNIMOD_PATTERN = re.compile(r"([A-Z])\(UniMod:\d+\)|([A-Z])")
_PSMTAG_NTERM = re.compile(r"^\.\[[\d.]+\]")
_PSMTAG_PATTERN = re.compile(r"([A-Z])\[[\d.]+\]|([A-Z])\(UniMod:\d+\)|([A-Z])")


def detect_nterm_mod(seq: str) -> int:
    m = _PSMTAG_NTERM.match(seq)
    if not m:
        return 0
    return NTERM_MOD_VOCAB.get(m.group(0), 0)


def detect_cterm_mod(seq: str) -> int:
    return 0  # no C-term α-carboxyl mod data yet


def tokenize_peptide(seq: str) -> list[int]:
    """Bracket-mod notation, e.g. `M[147]AAR` or `AAKC[160]R`. Unknown bracket mods
    collapse to the parent AA. `C` is always interpreted as camC (C[160]) in this
    format since it's derived from data where carbamidomethylation is fixed."""
    tokens: list[int] = []
    for m in _MOD_PATTERN.finditer(seq):
        full = m.group(0)
        aa = m.group(1) or m.group(2)
        if full.startswith("M["):
            tokens.append(VOCAB["M[147]"])
        elif aa == "C":
            tokens.append(VOCAB["C[160]"])
        elif aa in VOCAB:
            tokens.append(VOCAB[aa])
    return _pad(tokens)


def tokenize_unimod(seq: str) -> list[int]:
    """UniMod notation, e.g. `M(UniMod:35)AAR` or `AAKC(UniMod:4)R`. Plain C is
    also treated as camC in this format."""
    tokens: list[int] = []
    for m in _UNIMOD_PATTERN.finditer(seq):
        full = m.group(0)
        aa = m.group(1) or m.group(2)
        if aa == "M" and "(UniMod:35)" in full:
            tokens.append(VOCAB["M[147]"])
        elif aa == "C":
            tokens.append(VOCAB["C[160]"])
        elif aa in VOCAB:
            tokens.append(VOCAB[aa])
    return _pad(tokens)


def tokenize_psmtag(seq: str, use_k_tag: bool = True) -> list[int]:
    """FragPipe PSMtag format. N-term `.[mass]` is stripped (represented at
    peptide level via `nterm_mod_id`). `K[308.1161]` → dedicated `K[308]` token
    when `use_k_tag=True`; else falls back to plain K. `M(UniMod:35)` → oxM,
    `C(UniMod:4)` → camC."""
    seq = _PSMTAG_NTERM.sub("", seq)
    tokens: list[int] = []
    for m in _PSMTAG_PATTERN.finditer(seq):
        aa = m.group(1) or m.group(2) or m.group(3)
        full = m.group(0)
        if aa == "M" and ("[147]" in full or "(UniMod:35)" in full):
            tokens.append(VOCAB["M[147]"])
        elif aa == "C":
            tokens.append(VOCAB["C[160]"])
        elif aa == "K" and "[308.1161]" in full and use_k_tag:
            tokens.append(VOCAB["K[308]"])
        elif aa in VOCAB:
            tokens.append(VOCAB[aa])
    return _pad(tokens)


def _pad(tokens: list[int]) -> list[int]:
    if len(tokens) >= MAX_LEN:
        return tokens[:MAX_LEN]
    return tokens + [0] * (MAX_LEN - len(tokens))


def _pick_tokenizer(sample: str):
    if sample.startswith(".["):
        return tokenize_psmtag
    if "UniMod:" in sample:
        return tokenize_unimod
    return tokenize_peptide


# -------- Library loaders --------

def load_spectral_library(
    path: str, seed: int = 42, split_dir: str = "splits",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a DIA-NN spectral library TSV. Expects columns PeptideSequence /
    ModifiedPeptide / Tr_recalibrated. Groups by sequence (averaging Tr), then
    splits 80/10/10 at the peptide level."""
    df = pd.read_csv(path, sep="\t", usecols=["PeptideSequence", "ModifiedPeptide", "Tr_recalibrated"])
    df = df.dropna(subset=["Tr_recalibrated"]).rename(columns={"Tr_recalibrated": "Retention"})
    df["seq"] = df["ModifiedPeptide"].fillna(df["PeptideSequence"])
    df = df.dropna(subset=["seq"])
    df = df[df["seq"].str.len() > 0]
    df = df.groupby("seq", as_index=False)["Retention"].mean()
    return _peptide_split(df, seed, split_dir)


def load_psmtag_library(
    path: str, seed: int = 42, split_dir: str = "splits_psmtag",
    exclude_oxm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a FragPipe PSMtag library TSV. Expects columns PeptideSequence /
    ModifiedPeptideSequence / NormalizedRetentionTime."""
    df = pd.read_csv(path, sep="\t", usecols=[
        "PeptideSequence", "ModifiedPeptideSequence", "NormalizedRetentionTime",
    ])
    df = df.dropna(subset=["NormalizedRetentionTime"])
    df = df.rename(columns={"NormalizedRetentionTime": "Retention"})
    df["seq"] = df["ModifiedPeptideSequence"].fillna(df["PeptideSequence"])
    df = df.dropna(subset=["seq"])
    df = df[df["seq"].str.len() > 0]
    if exclude_oxm:
        df = df[~df["seq"].str.contains(r"M\(UniMod:35\)", regex=True)].reset_index(drop=True)
    df = df.groupby("seq", as_index=False)["Retention"].mean()
    return _peptide_split(df, seed, split_dir)


def _peptide_split(df: pd.DataFrame, seed: int, split_dir: str):
    unique_peptides = np.sort(df["seq"].unique())
    train_pep, temp_pep = train_test_split(unique_peptides, test_size=0.2, random_state=seed)
    val_pep, test_pep = train_test_split(temp_pep, test_size=0.5, random_state=seed)

    sp = Path(split_dir); sp.mkdir(parents=True, exist_ok=True)
    for name, peps in [("train", train_pep), ("val", val_pep), ("test", test_pep)]:
        pd.DataFrame({"seq": peps}).to_csv(sp / f"{name}_peptides.tsv", sep="\t", index=False)

    train_df = df[df["seq"].isin(set(train_pep))].reset_index(drop=True)
    val_df = df[df["seq"].isin(set(val_pep))].reset_index(drop=True)
    test_df = df[df["seq"].isin(set(test_pep))].reset_index(drop=True)
    return train_df, val_df, test_df


# -------- Dataset --------

class RTDataset(Dataset):
    """Holds tokenized peptide sequences + z-scored RT targets + terminal mod ids.

    __getitem__ returns (tokens, nterm_mod_id, cterm_mod_id, target) — a 4-tuple
    that matches the RTModel.forward signature. The target is z-scored against
    the provided rt_median / rt_mad (pass the TRAIN median/MAD when building
    val/test datasets to keep scales consistent).
    """

    def __init__(
        self, df: pd.DataFrame,
        rt_median: Optional[float] = None,
        rt_mad: Optional[float] = None,
    ):
        sample = df["seq"].iloc[0] if len(df) > 0 else ""
        tokenizer = _pick_tokenizer(sample)

        self.sequences = torch.tensor([tokenizer(s) for s in df["seq"]], dtype=torch.long)
        self.nterm_mod_ids = torch.tensor([detect_nterm_mod(s) for s in df["seq"]], dtype=torch.long)
        self.cterm_mod_ids = torch.tensor([detect_cterm_mod(s) for s in df["seq"]], dtype=torch.long)

        rts = df["Retention"].values.astype("float32")
        self.rt_median = float(rt_median if rt_median is not None else np.median(rts))
        self.rt_mad = float(rt_mad if rt_mad is not None else np.median(np.abs(rts - self.rt_median)))
        if self.rt_mad == 0:
            self.rt_mad = 1.0
        self.targets = torch.tensor((rts - self.rt_median) / self.rt_mad, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return (
            self.sequences[idx],
            self.nterm_mod_ids[idx],
            self.cterm_mod_ids[idx],
            self.targets[idx],
        )
