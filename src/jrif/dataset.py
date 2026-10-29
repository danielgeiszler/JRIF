"""Tokenization, vocabulary, and library loaders for JRIF.

The Vocabulary object owns the token-id mapping: residues, modified residues,
and terminus tokens. Every peptide is tokenized with a terminus token at both
ends — its N-terminal mod (e.g. `.[308]`) or `<nterm>` at position 0, and
`<cterm>` after the last residue — so the token sequence is the single place
termini are represented. The vocabulary is *runtime extensible*: when
fine-tuning on data containing tokens the base model has not seen,
Vocabulary.extend() grows the map and RTModel.extend_vocab grows the embedding
tables to match.

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
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
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
    # Free termini. An N-terminal mod token replaces <nterm>; a future
    # C-terminal mod token would replace <cterm> the same way.
    "<nterm>": 24,
    "<cterm>": 25,
}

FREE_NTERM = "<nterm>"
FREE_CTERM = "<cterm>"

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
    parent_of: dict[str, str] = field(default_factory=dict)
    fresh: set[str] = field(default_factory=set)

    @classmethod
    def default(cls) -> "Vocabulary":
        return cls(
            tokens=dict(DEFAULT_TOKENS),
            parent_of=dict(DEFAULT_PARENT_OF),
            fresh=set(DEFAULT_FRESH),
        )

    @property
    def size(self) -> int:
        return len(self.tokens)

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

    def parent_id(self, token: str) -> Optional[int]:
        parent = self.parent_of.get(token)
        if parent is None:
            return None
        return self.tokens.get(parent)

    def to_dict(self) -> dict:
        return {
            "tokens": dict(self.tokens),
            "parent_of": dict(self.parent_of),
            "fresh": sorted(self.fresh),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        return cls(
            tokens=dict(d["tokens"]),
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


class TokenizationError(ValueError):
    """A sequence could not be tokenized faithfully (unknown token/residue,
    malformed input, or truncation). Raised only in strict mode."""


def _pad(tokens: list[int]) -> list[int]:
    if len(tokens) >= MAX_LEN:
        return tokens[:MAX_LEN]
    return tokens + [0] * (MAX_LEN - len(tokens))


def tokenize(seq: str, vocab: Vocabulary, *, strict: bool = False) -> list[int]:
    """Tokenize one peptide sequence with the given Vocabulary.

    Returns token_ids[MAX_LEN]: N-terminus token (mod or <nterm>), residues,
    <cterm>, then padding. Tokens not in `vocab` fall back to the bare AA id
    when possible, otherwise are dropped.

    strict: raise TokenizationError instead of silently degrading — on
    non-string/empty input, unknown modifications (including fallback to the
    bare residue), unknown residues, unknown terminal mods, and sequences
    longer than MAX_LEN.
    """
    if not isinstance(seq, str) or not seq:
        if strict:
            raise TokenizationError(f"empty or non-string sequence: {seq!r}")
        return _pad([])
    body, nterm_mod = _strip_nterm(seq)
    fmt = _detect_format(seq)
    out: list[int] = []
    if nterm_mod:
        idx = vocab.tokens.get(nterm_mod)
        if idx is not None:
            out.append(idx)
        elif strict:
            raise TokenizationError(
                f"unknown N-terminal mod {nterm_mod!r} in {seq!r}")
    else:
        out.append(vocab.tokens[FREE_NTERM])
    for m in _iter_main_matches(body, fmt):
        aa = next((g for g in m.groups() if g), None)
        token = _resolve_main(aa, m.group(0), fmt)
        if token is None:
            if strict:
                raise TokenizationError(
                    f"unparseable token {m.group(0)!r} in {seq!r}")
            continue
        idx = vocab.tokens.get(token)
        if idx is None and strict:
            raise TokenizationError(f"unknown token {token!r} in {seq!r}")
        if idx is None and aa in vocab.tokens:
            idx = vocab.tokens[aa]
        if idx is None:
            continue
        out.append(idx)
    out.append(vocab.tokens[FREE_CTERM])
    if strict and len(out) > MAX_LEN:
        raise TokenizationError(
            f"sequence longer than MAX_LEN={MAX_LEN} ({len(out)} tokens): {seq!r}")
    return _pad(out)


def discover_tokens(
    seqs: Iterable[str], vocab: Vocabulary,
) -> dict[str, Optional[str]]:
    """Walk sequences and return tokens not yet present in `vocab`, mapped to
    their inferred parent AA (the bare residue letter for X[NNN]-form tokens;
    None otherwise, including N-terminal mods)."""
    new_main: dict[str, Optional[str]] = {}
    for seq in seqs:
        if not isinstance(seq, str) or not seq:
            continue
        body, nterm_mod = _strip_nterm(seq)
        if nterm_mod and nterm_mod not in vocab.tokens:
            new_main[nterm_mod] = None
        fmt = _detect_format(seq)
        for m in _iter_main_matches(body, fmt):
            aa = next((g for g in m.groups() if g), None)
            token = _resolve_main(aa, m.group(0), fmt)
            if token is None or token in vocab.tokens or token in new_main:
                continue
            parent = aa if aa and aa in vocab.tokens else None
            new_main[token] = parent
    return new_main


def strict_filter(
    df: pl.DataFrame, vocab: Vocabulary, *, seq_col: str = "seq",
    will_extend: bool = False,
) -> tuple[pl.DataFrame, int, Optional[str]]:
    """Drop rows whose sequence fails strict tokenization.

    Returns (filtered_df, n_dropped, first_failure_message). When
    `will_extend` is set, tokens that `discover_tokens` would report are
    treated as known — the caller is about to extend the vocabulary with
    them — so only genuinely untokenizable rows (unparseable characters,
    over-MAX_LEN sequences, ...) are dropped.
    """
    seqs = df[seq_col].to_list()
    check_vocab = vocab
    if will_extend:
        new_main = discover_tokens(seqs, vocab)
        if new_main:
            check_vocab = Vocabulary(tokens=dict(vocab.tokens))
            for t in new_main:
                check_vocab.extend(t, None)
    keep = np.ones(len(seqs), dtype=bool)
    first: Optional[str] = None
    for i, s in enumerate(seqs):
        try:
            tokenize(s, check_vocab, strict=True)
        except TokenizationError as e:
            keep[i] = False
            if first is None:
                first = str(e)
    n_bad = int((~keep).sum())
    if n_bad:
        df = df.filter(pl.Series(keep))
    return df, n_bad, first


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


class RTDataset(Dataset):
    """Tokenized peptides + z-scored RT targets.

    __getitem__ returns (tokens, target). The
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
        strict: bool = False,
    ):
        df = to_polars(df)
        seqs = df[seq_col].to_list()
        n = len(seqs)

        tok_arr = np.zeros((n, MAX_LEN), dtype=np.int64)
        for i, s in enumerate(seqs):
            tok_arr[i, :] = tokenize(s, vocab, strict=strict)
        self.sequences = torch.from_numpy(tok_arr)

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
        return self.sequences[idx], self.targets[idx]
