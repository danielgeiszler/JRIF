"""Tokenization, vocabulary, and dataset helpers for JRIF.

The Vocabulary object owns the token-id mapping: residues, modified residues,
and terminus tokens. Every peptide is tokenized with a terminus token at both
ends — its N-terminal mod or `<nterm>` at position 0, and its C-terminal mod
or `<cterm>` after the last residue — so the token sequence is the single
place termini are represented. The vocabulary is *runtime extensible*: when
fine-tuning on data containing tokens the base model has not seen,
Vocabulary.extend() grows the map and RTModel.extend_vocab grows the embedding
tables to match.

Modification notation:
- Every modified token is `<site>[<signed delta mass, 4 decimals>]`, where
  site is the residue letter, `n` (N-terminus) or `c` (C-terminus):
  `C[+57.0215]`, `M[+15.9949]`, `K[+308.1161]`, `n[+308.1161]`.
- A bracket mass is the modification's delta at the precision it was
  written: `S[80]` and `S[+80]` -> `S[+80]`; `S[+79.9663]` stays. Known
  chemistry — `(UniMod:n)` in any casing, via the bundled UniMod table, and
  LEGACY_ALIASES — is named at 4 decimals.
- LEGACY_ALIASES are the established spellings of the shipped models' mods:
  `C[160]` -> `C[+57.0215]`, `M[147]` -> `M[+15.9949]`, `K[308]` ->
  `K[+308.1161]`, `.[308]` -> `n[+308.1161]`.
- A written mod matches a vocabulary token when the two agree at the coarser
  of their precisions (`S[80]` matches `S[+79.9663]`). A vocabulary never
  holds two tokens on one site that agree that way, and a spelling matching
  more than one token raises instead of picking.
- Bare residues mean exactly that: a bare `C` is unmodified cysteine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
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
    "M[+15.9949]": 21,
    "C[+57.0215]": 22,
    "K[+308.1161]": 23,
    # Free termini. An N-terminal mod token replaces <nterm>; a future
    # C-terminal mod token would replace <cterm> the same way.
    "<nterm>": 24,
    "<cterm>": 25,
}

FREE_NTERM = "<nterm>"
FREE_CTERM = "<cterm>"

DEFAULT_PARENT_OF: dict[str, str] = {
    "M[+15.9949]": "M",
    "C[+57.0215]": "C",
    "K[+308.1161]": "K",
}

# K[+308.1161]'s side-chain tag changes chemistry too much to inherit K's physico
# features. Only the learned embedding picks up signal for it.
DEFAULT_FRESH: set[str] = {"K[+308.1161]"}

CANONICAL_AAS: tuple[str, ...] = (
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
)

# The limited harmonization: the legacy unsigned spellings of the shipped
# models' modifications, keyed by (site, integer part of the written mass).
# These were written as residue+delta (C[160], M[147]) or as a bare tag mass
# (K[308], .[308]); every other bracket mass is taken as a delta as written.
LEGACY_ALIASES: dict[tuple[str, int], float] = {
    ("C", 160): 57.021464,   # carbamidomethyl, FragPipe absolute
    ("M", 147): 15.994915,   # oxidation, FragPipe absolute
    ("K", 308): 308.1161,    # PSMtag side-chain tag
    ("n", 308): 308.1161,    # PSMtag N-terminal tag
}


def _load_unimod() -> dict[int, float]:
    path = Path(__file__).with_name("unimod.tsv")
    rows = (l.split("\t") for l in path.read_text().splitlines()
            if l and not l.startswith(("#", "accession")))
    return {int(acc): float(mass) for acc, _, mass in rows}


# UniMod accession -> monoisotopic delta (full table, see unimod.tsv).
UNIMOD_DELTA: dict[int, float] = _load_unimod()

_RESIDUES = frozenset(CANONICAL_AAS)
_MOD_AT = re.compile(
    r"\[(?P<sign>[+-]?)(?P<mass>\d+(?:\.\d*)?|\.\d+)\]|\((?i:unimod):(?P<um>\d+)\)"
)
_MOD_NAME = re.compile(r"^(?P<site>[A-Zcn])\[(?P<mass>[+-]\d+(?:\.\d+)?)\]$")

# Known chemistry (UniMod table, LEGACY_ALIASES) is named at this precision.
KNOWN_MASS_DECIMALS = 4


class TokenizationError(ValueError):
    """A sequence could not be tokenized faithfully (unknown token/residue,
    malformed input, ambiguous mass, or over-length)."""


def mod_token(site: str, mass: Decimal) -> str:
    """Token name for `mass` on `site` (residue letter, `n` or `c`), keeping
    exactly the mass's precision: Decimal("80") -> `S[+80]`."""
    return f"{site}[{mass:+f}]"


def _mod_key(name: str) -> Optional[tuple[str, Decimal]]:
    m = _MOD_NAME.match(name)
    return (m.group("site"), Decimal(m.group("mass"))) if m else None


def _decimals(mass: Decimal) -> int:
    return max(0, -mass.as_tuple().exponent)


def _agree(a: Decimal, b: Decimal) -> bool:
    """Equal at the coarser of the two precisions (half-up rounding)."""
    q = Decimal(1).scaleb(-min(_decimals(a), _decimals(b)))
    return a.quantize(q, ROUND_HALF_UP) == b.quantize(q, ROUND_HALF_UP)


@dataclass
class Vocabulary:
    """Token-id maps + lineage tracking. Mutate via extend()."""
    tokens: dict[str, int]
    parent_of: dict[str, str] = field(default_factory=dict)
    fresh: set[str] = field(default_factory=set)
    # site -> [(token, mass)] and spelling -> id caches; rebuilt after extend()
    _by_site: Optional[dict] = field(default=None, init=False, repr=False, compare=False)
    _ids: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def default(cls) -> "Vocabulary":
        return cls(
            tokens=dict(DEFAULT_TOKENS),
            parent_of=dict(DEFAULT_PARENT_OF),
            fresh=set(DEFAULT_FRESH),
        )

    def copy(self) -> "Vocabulary":
        return Vocabulary(dict(self.tokens), dict(self.parent_of), set(self.fresh))

    @property
    def size(self) -> int:
        return len(self.tokens)

    def has(self, token: str) -> bool:
        return token in self.tokens

    def matches(self, name: str) -> list[str]:
        """Vocabulary mod tokens a written mod `name` agrees with."""
        key = _mod_key(name)
        if key is None:
            return []
        if self._by_site is None:
            self._by_site = {}
            for t in self.tokens:
                k = _mod_key(t)
                if k is not None:
                    self._by_site.setdefault(k[0], []).append((t, k[1]))
        site, mass = key
        return [t for t, m in self._by_site.get(site, ()) if _agree(m, mass)]

    def match(self, name: str) -> Optional[str]:
        """The vocabulary token for a written token, or None if it has none.
        Raises TokenizationError when a mod agrees with several tokens."""
        if name in self.tokens:
            return name
        hits = self.matches(name)
        if len(hits) > 1:
            raise TokenizationError(
                f"{name} matches several model tokens ({', '.join(sorted(hits))}); "
                f"write the mass with more precision")
        return hits[0] if hits else None

    def id_of(self, name: str) -> Optional[int]:
        """Token id for a written token (see match); cached per spelling."""
        hit = self._ids.get(name)
        if hit is None:
            try:
                tok = self.match(name)
                hit = self.tokens[tok] if tok is not None else None
            except TokenizationError as e:
                hit = e
            self._ids[name] = hit
        if isinstance(hit, TokenizationError):
            raise hit
        return hit

    def extend(self, token: str, parent_aa: Optional[str], fresh: bool = False) -> int:
        if token in self.tokens:
            return self.tokens[token]
        if "[" in token and _mod_key(token) is None:
            raise ValueError(f"{token!r} is not a token name; use canonical_token()")
        clash = self.matches(token)
        if clash:
            raise ValueError(f"cannot add {token}: it agrees with existing "
                             f"token(s) {', '.join(sorted(clash))}")
        new_id = len(self.tokens)
        self.tokens[token] = new_id
        if parent_aa is not None and not fresh:
            self.parent_of[token] = parent_aa
        if fresh:
            self.fresh.add(token)
        self._by_site = None
        self._ids = {}
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

def _site_label(site: str) -> str:
    return {"n": "N-terminal", "c": "C-terminal"}.get(site, f"residue {site}")


def _known(value: float) -> Decimal:
    return Decimal(f"{value:.{KNOWN_MASS_DECIMALS}f}")


def _resolve_mass(site: str, m: re.Match) -> Decimal:
    if m.group("um") is not None:
        acc = int(m.group("um"))
        if acc not in UNIMOD_DELTA:
            raise TokenizationError(
                f"UniMod:{acc} on {_site_label(site)} is not a known UniMod "
                f"accession; write the delta mass instead, e.g. {site}[+12.3456]")
        return _known(UNIMOD_DELTA[acc])
    mass = Decimal(m.group("mass"))
    if m.group("sign") == "-":
        return -mass
    if not m.group("sign"):
        alias = LEGACY_ALIASES.get((site, int(mass)))
        if alias is not None:
            return _known(alias)
    return mass


def _mod_or_error(site: str, m: re.Match) -> "str | TokenizationError":
    try:
        return mod_token(site, _resolve_mass(site, m))
    except TokenizationError as e:
        return e


@dataclass
class _Parsed:
    """One peptide split into terminal and residue tokens, each named with
    its written precision. A position whose annotation could not be
    resolved holds the TokenizationError instead; each residue is paired
    with its bare letter for the lenient fallback. `raw` maps each mod
    token to the text the user wrote for it, for logs and errors."""
    nterm: "str | TokenizationError | None" = None
    residues: list[tuple[str, "str | TokenizationError"]] = field(default_factory=list)
    cterm: "str | TokenizationError | None" = None
    errors: list[TokenizationError] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)

    def spelled(self, token: str) -> str:
        return self.raw.get(token, token)


def _parse(seq: str) -> _Parsed:
    """Accepted syntax: residues `A`..`Y`, each optionally followed by
    `[mass]` or `(UniMod:n)`; an N-terminal mod as a leading `.[m]`, `n[m]`,
    `[m]-` or `(UniMod:n)`; a C-terminal mod as a trailing `c[m]` or `-[m]`.
    Unrecognized characters are reported in `errors`, never silently kept."""
    out = _Parsed()
    i, n = 0, len(seq)

    def mod(site: str, m: re.Match, start: int, end: int):
        t = _mod_or_error(site, m)
        if isinstance(t, str):
            out.raw.setdefault(t, seq[start:end])
        return t

    if seq[:2] in (".[", "n["):
        m = _MOD_AT.match(seq, 1)
        if m:
            out.nterm, i = mod("n", m, 0, m.end()), m.end()
    elif seq[:1] in ("[", "("):
        m = _MOD_AT.match(seq, 0)
        if m and (m.group("um") is not None or seq[m.end():m.end() + 1] == "-"):
            i = m.end() + (m.group("um") is None)
            out.nterm = mod("n", m, 0, i)
    while i < n:
        ch = seq[i]
        if ch in _RESIDUES:
            m = _MOD_AT.match(seq, i + 1)
            if m:
                out.residues.append((ch, mod(ch, m, i, m.end())))
                i = m.end()
            else:
                out.residues.append((ch, ch))
                i += 1
            continue
        if ch in "c-":
            m = _MOD_AT.match(seq, i + 1)
            if m and m.end() == n:
                out.cterm, i = mod("c", m, i, n), n
                continue
        out.errors.append(TokenizationError(
            f"unparseable character {ch!r} at position {i}"))
        i += 1
    for t in (out.nterm, out.cterm, *(tok for _, tok in out.residues)):
        if isinstance(t, TokenizationError):
            out.errors.append(t)
    return out


def canonical_token(token: str) -> str:
    """Token name for a single token in any accepted notation, precision as
    written: `S[80]` -> `S[+80]`, `K[308]` -> `K[+308.1161]`,
    `.[308]` -> `n[+308.1161]`, `C(UniMod:4)` -> `C[+57.0215]`."""
    p = _parse(token)
    if p.errors:
        raise p.errors[0]
    parts = [t for t in (p.nterm, *(tok for _, tok in p.residues), p.cterm) if t]
    if len(parts) != 1:
        raise TokenizationError(f"expected a single token, got {token!r}")
    return parts[0]


def tokenize(seq: str, vocab: Vocabulary) -> list[int]:
    """Tokenize one peptide sequence with the given Vocabulary.

    Returns token_ids[MAX_LEN]: N-terminus token (mod or <nterm>), residues,
    C-terminus token (mod or <cterm>), then padding. See the module docstring
    for the accepted notation and how written mods match vocabulary tokens.

    The sequence is represented exactly or not at all: TokenizationError on
    non-string/empty input, unparseable characters, unknown UniMod
    accessions, mods matching no (or several) vocabulary tokens, and
    sequences longer than MAX_LEN.
    """
    if not isinstance(seq, str) or not seq:
        raise TokenizationError(f"empty or non-string sequence: {seq!r}")
    p = _parse(seq)
    if p.errors:
        raise TokenizationError(f"{p.errors[0]} in {seq!r}")

    def lookup(token: str, label: str) -> int:
        try:
            idx = vocab.id_of(token)
        except TokenizationError as e:  # ambiguous: re-quote the user's spelling
            raise TokenizationError(
                f"{p.spelled(token)!r} in {seq!r} {str(e).split(' ', 1)[1]}") from None
        if idx is None:
            raise TokenizationError(f"unknown {label} {p.spelled(token)!r} in {seq!r}")
        return idx

    out = [lookup(p.nterm, "N-terminal mod") if p.nterm else vocab.tokens[FREE_NTERM]]
    out += [lookup(token, "modification") for _, token in p.residues]
    out.append(lookup(p.cterm, "C-terminal mod") if p.cterm else vocab.tokens[FREE_CTERM])
    if len(out) > MAX_LEN:
        raise TokenizationError(
            f"sequence longer than MAX_LEN={MAX_LEN} ({len(out)} tokens): {seq!r}")
    return out + [0] * (MAX_LEN - len(out))


def _written_mods(seqs: Iterable[str]) -> dict[str, tuple[int, str, Optional[str]]]:
    """Every mod in `seqs`, keyed by the text as written -> (n sequences
    containing it, token name, parent residue or None for terminal mods)."""
    found: dict[str, tuple[int, str, Optional[str]]] = {}
    for seq in seqs:
        if not isinstance(seq, str) or not seq:
            continue
        p = _parse(seq)
        here = {t: None for t in (p.nterm, p.cterm) if isinstance(t, str)}
        here.update({t: aa for aa, t in p.residues if isinstance(t, str) and t != aa})
        for t, parent in here.items():
            raw = p.spelled(t)
            found[raw] = (found.get(raw, (0,))[0] + 1, t, parent)
    return found


def discover_tokens(
    seqs: Iterable[str], vocab: Vocabulary,
) -> dict[str, Optional[str]]:
    """Mods in `seqs` that match no `vocab` token, mapped to their parent
    residue (None for terminal mods). Spellings that agree with each other
    become one token, named by the most precise spelling. Spellings that
    are ambiguous or fail to parse are skipped; `strict_filter` reports them."""
    names = {t: parent for _, t, parent in _written_mods(seqs).values()}
    trial = vocab.copy()
    new: dict[str, Optional[str]] = {}
    for t in sorted(names, key=lambda t: (-_decimals(_mod_key(t)[1]), t)):
        try:
            if trial.match(t) is None:
                trial.extend(t, names[t])
                new[t] = names[t]
        except TokenizationError:
            continue
    return new


def mod_census(
    seqs: Iterable[str], vocab: Vocabulary,
) -> list[tuple[str, int, "str | None", bool, Optional[str]]]:
    """Every modification in `seqs`, as (spelling, n_ids, token, is_new,
    parent), sorted by spelling. spelling is the text exactly as written;
    token is the vocabulary token it resolves to once new tokens are added
    (None if ambiguous); is_new is relative to `vocab`; parent is the
    residue a new token inherits from (None: learned from scratch)."""
    seqs = list(seqs)
    new = discover_tokens(seqs, vocab)
    trial = vocab.copy()
    for t, parent in new.items():
        trial.extend(t, parent)
    out = []
    for raw, (n, t, parent) in sorted(_written_mods(seqs).items()):
        try:
            tok = trial.match(t)
        except TokenizationError:
            tok = None
        out.append((raw, n, tok, tok in new, new[tok] if tok in new else parent))
    return out


def drop_untokenizable(
    df: pl.DataFrame, vocab: Vocabulary, *, seq_col: str = "seq",
    will_extend: bool = False,
) -> tuple[pl.DataFrame, int, Optional[str]]:
    """Drop rows whose sequence cannot be tokenized.

    Returns (filtered_df, n_dropped, first_failure_message). When
    `will_extend` is set, tokens that `discover_tokens` would report are
    treated as known — the caller is about to extend the vocabulary with
    them — so only genuinely untokenizable rows (unparseable characters,
    ambiguous masses, over-MAX_LEN sequences, ...) are dropped.
    """
    seqs = df[seq_col].to_list()
    check_vocab = vocab
    if will_extend:
        new = discover_tokens(seqs, vocab)
        if new:
            check_vocab = vocab.copy()
            for t, parent in new.items():
                check_vocab.extend(t, parent)
    keep = np.ones(len(seqs), dtype=bool)
    first: Optional[str] = None
    for i, s in enumerate(seqs):
        try:
            tokenize(s, check_vocab)
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
    ):
        df = to_polars(df)
        seqs = df[seq_col].to_list()
        n = len(seqs)

        tok_arr = np.zeros((n, MAX_LEN), dtype=np.int64)
        for i, s in enumerate(seqs):
            tok_arr[i, :] = tokenize(s, vocab)
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
