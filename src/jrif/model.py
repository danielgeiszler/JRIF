"""Dilated-CNN retention-time model.

The model is *runtime extensible*: its embedding tables and physico/global
lookup buffers are sized from a Vocabulary at construction time, and the
RTModel.extend_vocab() method grows them in-place when fine-tuning encounters
new modification tokens. New-token rows are initialized by inheriting from
the parent residue's row (or zero, for tokens flagged as `fresh` because their
chemistry diverges too far — K[308]'s side-chain tag is the canonical example).

Architecture overview:
  Token indices -> [physicochemical features (12) + learned embedding (D)]
    -> 5 dilated Conv1d blocks (k=3; dilations 1,2,4,8,16; GELU + BN)
    -> masked GAP -> [128 ch]
  Head input concatenates:
    GAP (128) + n_term_emb (4) + c_term_emb (4) + n_mod (4) + c_mod (4)
    + GRAVY (1) + net charge @ pH 2.7 (1) + frac aromatic (1)
    + normalized seq length (1) + frac hydrophobic (1) = 149 dims
  FC head: Dropout(0.3) -> Linear(149, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from jrif.dataset import Vocabulary, MAX_LEN, CANONICAL_AAS


# Per-canonical-AA raw physico features and global scalars (order matches CANONICAL_AAS).
_PHYSICO_RAW = torch.tensor([
    [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3],
    [71.04, 103.01, 115.03, 129.04, 147.07, 57.02, 137.06, 113.08, 128.09, 113.08, 131.04, 114.04, 97.05, 128.06, 156.10, 87.03, 101.05, 99.07, 186.08, 163.06],
    [0.0, 8.3, 3.65, 4.25, 0.0, 0.0, 6.0, 0.0, 10.5, 0.0, 0.0, 0.0, 0.0, 0.0, 12.5, 0.0, 0.0, 0.0, 0.0, 10.1],
    [67.0, 86.0, 91.0, 109.0, 135.0, 48.0, 118.0, 124.0, 135.0, 124.0, 124.0, 96.0, 90.0, 114.0, 148.0, 73.0, 93.0, 105.0, 163.0, 141.0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 2, 4, 1, 1, 0, 1, 1],
    [0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 1, 1, 0, 0, 1],
    [6.00, 5.07, 2.77, 3.22, 5.48, 5.97, 7.59, 6.02, 9.74, 5.98, 5.74, 5.41, 6.30, 5.65, 10.76, 5.68, 5.60, 5.96, 5.89, 5.66],
    [1.42, 0.70, 1.01, 1.51, 1.13, 0.57, 1.00, 1.08, 1.16, 1.21, 1.45, 0.67, 0.57, 1.11, 0.98, 0.77, 0.83, 1.06, 1.08, 0.69],
    [0.83, 1.19, 0.54, 0.37, 1.38, 0.75, 0.87, 1.60, 0.74, 1.30, 1.05, 0.89, 0.55, 1.10, 0.93, 0.75, 1.19, 1.70, 1.37, 1.47],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
], dtype=torch.float32)  # shape (12, 20)

_KD = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]
_CHARGE_27 = [0, 0, 0.9, 0.97, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0]
_AROMATIC = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
_HYDROPHOBIC = [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0]


def _z_score_per_feature(matrix: torch.Tensor) -> torch.Tensor:
    """Z-score each feature row across the AA dimension."""
    mean = matrix.mean(dim=1, keepdim=True)
    std = matrix.std(dim=1, keepdim=True).clamp(min=1e-6)
    return (matrix - mean) / std


def _parent_aa_for(token: str, vocab: Vocabulary) -> Optional[str]:
    """Return the canonical AA letter to inherit from, or None if the token
    inherits nothing (pad / fresh / no parent)."""
    if token == "<pad>":
        return None
    if token in CANONICAL_AAS:
        return token
    if token in vocab.fresh:
        return None
    return vocab.parent_of.get(token)


def _build_physico_matrix(vocab: Vocabulary) -> torch.Tensor:
    raw = _z_score_per_feature(_PHYSICO_RAW)  # (12, 20)
    out = torch.zeros(vocab.size, 12)
    for token, idx in vocab.tokens.items():
        parent = _parent_aa_for(token, vocab)
        if parent is None:
            continue
        out[idx] = raw[:, CANONICAL_AAS.index(parent)]
    return out


def _build_global_tables(vocab: Vocabulary) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hydro = torch.zeros(vocab.size)
    charge = torch.zeros(vocab.size)
    aromatic = torch.zeros(vocab.size)
    hydrophobic = torch.zeros(vocab.size)
    for token, idx in vocab.tokens.items():
        parent = _parent_aa_for(token, vocab)
        if parent is None:
            continue
        p = CANONICAL_AAS.index(parent)
        hydro[idx] = _KD[p]
        charge[idx] = _CHARGE_27[p]
        aromatic[idx] = _AROMATIC[p]
        hydrophobic[idx] = _HYDROPHOBIC[p]
    return hydro, charge, aromatic, hydrophobic


def _resize_embedding(
    emb: nn.Embedding, new_size: int, init_rows: torch.Tensor,
    *, padding_idx: Optional[int] = None,
) -> nn.Embedding:
    """Replace `emb` with a larger embedding whose first rows are `emb`'s
    weights and whose trailing rows are `init_rows`. init_rows.shape =
    (new_size - emb.num_embeddings, emb.embedding_dim)."""
    old = emb.weight.data
    assert init_rows.size(0) == new_size - old.size(0)
    assert init_rows.size(1) == old.size(1)
    new_weight = torch.cat([old, init_rows.to(old.device, old.dtype)], dim=0)
    new_emb = nn.Embedding(new_size, emb.embedding_dim, padding_idx=padding_idx)
    new_emb.to(old.device)
    new_emb.weight.data.copy_(new_weight)
    return new_emb


class DilatedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RTModel(nn.Module):
    """Dilated-CNN retention-time model. Sized from a Vocabulary; grow at
    runtime via `extend_vocab`."""

    def __init__(self, vocab: Vocabulary, embedding_dim: int = 16):
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim

        self._register_lookup_buffers(vocab)

        self.embedding = nn.Embedding(vocab.size, embedding_dim, padding_idx=0)
        self.n_term_embedding = nn.Embedding(vocab.size, 4, padding_idx=0)
        self.c_term_embedding = nn.Embedding(vocab.size, 4, padding_idx=0)
        self.nterm_mod_embedding = nn.Embedding(max(vocab.nterm_size, 1), 4)
        self.cterm_mod_embedding = nn.Embedding(max(vocab.cterm_size, 1), 4)

        conv_in = 12 + embedding_dim
        self.convs = nn.Sequential(
            DilatedBlock(conv_in, 64, kernel_size=3, dilation=1),
            DilatedBlock(64, 64, kernel_size=3, dilation=2),
            DilatedBlock(64, 128, kernel_size=3, dilation=4),
            DilatedBlock(128, 128, kernel_size=3, dilation=8),
            DilatedBlock(128, 128, kernel_size=3, dilation=16),
        )

        head_in = 128 + 4 + 4 + 4 + 4 + 3 + 1 + 1   # = 149
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    # ---- lookup buffers ----

    def _register_lookup_buffers(self, vocab: Vocabulary) -> None:
        self.register_buffer("physico_features", _build_physico_matrix(vocab), persistent=False)
        h, c, a, hd = _build_global_tables(vocab)
        self.register_buffer("hydro_table", h, persistent=False)
        self.register_buffer("charge_table", c, persistent=False)
        self.register_buffer("aromatic_table", a, persistent=False)
        self.register_buffer("hydrophobic_table", hd, persistent=False)

    def _refresh_lookup_buffers(self) -> None:
        device = self.physico_features.device
        self.physico_features = _build_physico_matrix(self.vocab).to(device)
        h, c, a, hd = _build_global_tables(self.vocab)
        self.hydro_table = h.to(device)
        self.charge_table = c.to(device)
        self.aromatic_table = a.to(device)
        self.hydrophobic_table = hd.to(device)

    # ---- runtime extension ----

    def extend_vocab(
        self,
        new_main_tokens: Iterable[tuple[str, Optional[str], bool]] = (),
        new_nterm_mods: Iterable[str] = (),
        new_cterm_mods: Iterable[str] = (),
    ) -> None:
        """Grow the vocab + every dependent embedding table / lookup buffer
        in-place. After calling this, any optimizer holding references to the
        old embedding parameters MUST be rebuilt.

        new_main_tokens: iterable of (token, parent_aa | None, fresh: bool).
        Parent's row is copied for the new main / n-term / c-term embedding
        rows; physico and global lookup rows are recomputed from vocab lineage.
        """
        new_main = list(new_main_tokens)
        new_nterm = list(new_nterm_mods)
        new_cterm = list(new_cterm_mods)
        if not new_main and not new_nterm and not new_cterm:
            return

        device = self.embedding.weight.device

        # Precompute init rows for main / terminal embeddings using the OLD vocab.
        if new_main:
            n_new = len(new_main)
            main_init = torch.zeros(n_new, self.embedding.embedding_dim, device=device)
            nterm_aa_init = torch.zeros(n_new, self.n_term_embedding.embedding_dim, device=device)
            cterm_aa_init = torch.zeros(n_new, self.c_term_embedding.embedding_dim, device=device)
            for k, (_, parent_aa, fresh) in enumerate(new_main):
                if parent_aa is None or fresh:
                    continue
                pid = self.vocab.tokens.get(parent_aa)
                if pid is None:
                    continue
                main_init[k] = self.embedding.weight.data[pid].clone()
                nterm_aa_init[k] = self.n_term_embedding.weight.data[pid].clone()
                cterm_aa_init[k] = self.c_term_embedding.weight.data[pid].clone()

            for token, parent_aa, fresh in new_main:
                self.vocab.extend(token, parent_aa, fresh)

            self.embedding = _resize_embedding(self.embedding, self.vocab.size, main_init, padding_idx=0)
            self.n_term_embedding = _resize_embedding(self.n_term_embedding, self.vocab.size, nterm_aa_init, padding_idx=0)
            self.c_term_embedding = _resize_embedding(self.c_term_embedding, self.vocab.size, cterm_aa_init, padding_idx=0)
            self._refresh_lookup_buffers()

        if new_nterm:
            init = torch.zeros(len(new_nterm), self.nterm_mod_embedding.embedding_dim, device=device)
            for mod in new_nterm:
                self.vocab.extend_nterm_mod(mod)
            self.nterm_mod_embedding = _resize_embedding(
                self.nterm_mod_embedding, self.vocab.nterm_size, init,
            )

        if new_cterm:
            init = torch.zeros(len(new_cterm), self.cterm_mod_embedding.embedding_dim, device=device)
            for mod in new_cterm:
                self.vocab.extend_cterm_mod(mod)
            self.cterm_mod_embedding = _resize_embedding(
                self.cterm_mod_embedding, self.vocab.cterm_size, init,
            )

    # ---- forward ----

    def forward(
        self,
        x: torch.Tensor,
        nterm_mod_id: Optional[torch.Tensor] = None,
        cterm_mod_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.size(0)
        tokens = x
        mask = (x != 0).unsqueeze(1)
        token_mask = (tokens != 0).float()

        n_term_token = tokens[:, 0]
        lengths = (x != 0).sum(dim=1)
        c_term_idx = (lengths - 1).clamp(min=0)
        c_term_tokens = x.gather(1, c_term_idx.unsqueeze(1)).squeeze(1)
        n_term_emb = self.n_term_embedding(n_term_token)
        c_term_emb = self.c_term_embedding(c_term_tokens)

        if nterm_mod_id is None:
            nterm_mod_id = torch.zeros(B, dtype=torch.long, device=tokens.device)
        if cterm_mod_id is None:
            cterm_mod_id = torch.zeros(B, dtype=torch.long, device=tokens.device)
        n_mod = self.nterm_mod_embedding(nterm_mod_id)
        c_mod = self.cterm_mod_embedding(cterm_mod_id)

        phys = self.physico_features[x]
        emb = self.embedding(x)
        z = torch.cat([phys, emb], dim=2).permute(0, 2, 1)
        z = self.convs(z)
        z = (z * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)

        seq_lengths = token_mask.sum(dim=1, keepdim=True).clamp(min=1)
        gravy = (self.hydro_table[tokens] * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        net_charge = (self.charge_table[tokens] * token_mask).sum(dim=1, keepdim=True)
        frac_aromatic = (self.aromatic_table[tokens] * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        frac_hydrophobic = (self.hydrophobic_table[tokens] * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        norm_seq_len = seq_lengths / float(MAX_LEN)

        feats = torch.cat([
            z, n_term_emb, c_term_emb, n_mod, c_mod,
            gravy, net_charge, frac_aromatic,
            norm_seq_len, frac_hydrophobic,
        ], dim=1)
        return self.head(feats).squeeze(1)


def load_legacy_state_dict(model: RTModel, state_dict: dict) -> None:
    """Load a pre-format-version checkpoint into a fresh RTModel built with
    Vocabulary.default(). Legacy checkpoints have an over-sized
    nterm_mod_embedding / cterm_mod_embedding (capacity 8 rather than the
    actual 2 / 1 used). Slice their weights to fit; load everything else as-is.
    """
    target = model.state_dict()
    new_state: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        if key not in target:
            continue
        tshape = target[key].shape
        if val.shape == tshape:
            new_state[key] = val
        elif val.dim() == target[key].dim() and all(v >= t for v, t in zip(val.shape, tshape)):
            slices = tuple(slice(0, s) for s in tshape)
            new_state[key] = val[slices].contiguous()
        else:
            # incompatible — keep model's init
            new_state[key] = target[key]
    model.load_state_dict(new_state, strict=False)
