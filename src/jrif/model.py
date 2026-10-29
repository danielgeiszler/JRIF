"""Dilated-CNN retention-time model.

The model is *runtime extensible*: its embedding tables and physico/global
lookup buffers are sized from a Vocabulary at construction time, and the
RTModel.extend_vocab() method grows them in-place when fine-tuning encounters
new modification tokens. New-token rows are initialized by inheriting from
the parent residue's row (or zero, for tokens flagged as `fresh` because their
chemistry diverges too far — K[308]'s side-chain tag is the canonical example).

Architecture overview:
  Token indices (termini included as tokens, see dataset.py)
    -> [physicochemical features (12) + learned embedding (D)]
    -> 5 dilated Conv1d blocks (k=3; dilations 1,2,4,8,16; GELU + BN)
    -> masked GAP -> [128 ch]
  Head input concatenates:
    GAP (128) + GRAVY (1) + net charge @ pH 2.7 (1) + frac aromatic (1)
    + normalized seq length (1) + frac hydrophobic (1) = 133 dims
  FC head: Dropout(0.3) -> Linear(133, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Return the canonical AA letter whose physico/global features the token
    inherits, or None (pad / fresh / terminal tokens, whose parent is
    <nterm>/<cterm> and which carry no residue features)."""
    if token == "<pad>":
        return None
    if token in CANONICAL_AAS:
        return token
    if token in vocab.fresh:
        return None
    parent = vocab.parent_of.get(token)
    return parent if parent in CANONICAL_AAS else None


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


def _conv1d_gemm(
    x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor],
    d: int, k: int, p: int,
) -> torch.Tensor:
    """k-tap dilated 1d conv as a single (C_out, k*C_in) GEMM."""
    xp = F.pad(x, (p, p)) if p else x
    L_out = xp.shape[2] - (k - 1) * d
    # Stack the k shifted views on the channel dim -> one (Cout, k*Cin) GEMM.
    xcat = torch.cat([xp[:, :, j * d : j * d + L_out] for j in range(k)], dim=1)
    wf = torch.cat([w[:, :, j] for j in range(k)], dim=1)
    out = torch.einsum("oc,bcl->bol", wf, xcat)
    if b is not None:
        out = out + b.view(1, -1, 1)
    return out


def fast_conv1d(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """Apply `conv` to x, decomposing into a single GEMM on CPU.

    PyTorch's CPU dispatch sends dilated (and some plain) Conv1d shapes to a
    naive `slow_conv_dilated2d` fallback with no fast kernel — profiled at
    ~68% of total train time. A k-tap conv is exactly the sum of k shifted
    (C_out, C_in) matmuls, fused here into one BLAS GEMM. GPU backends have
    proper conv kernels, so they keep the native path.
    """
    if x.device.type != "cpu":
        return conv(x)
    return _conv1d_gemm(
        x, conv.weight, conv.bias,
        conv.dilation[0], conv.kernel_size[0], conv.padding[0],
    )


def fast_conv_bn1d(x: torch.Tensor, conv: nn.Conv1d, bn: nn.BatchNorm1d) -> torch.Tensor:
    """conv + BatchNorm1d with BN folded into the conv weights.

    Exact at eval time (BN uses frozen running stats, an affine transform that
    commutes into the conv). Callers must only use this when NOT training.
    """
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale.view(-1, 1, 1)
    b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    b = bn.bias + (b0 - bn.running_mean) * scale
    if x.device.type != "cpu":
        return F.conv1d(x, w, b, padding=conv.padding[0], dilation=conv.dilation[0])
    return _conv1d_gemm(x, w, b, conv.dilation[0], conv.kernel_size[0], conv.padding[0])


class DilatedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.act(self.bn(fast_conv1d(x, self.conv)))
        return self.act(fast_conv_bn1d(x, self.conv, self.bn))


class RTModel(nn.Module):
    """Dilated-CNN retention-time model. Sized from a Vocabulary; grow at
    runtime via `extend_vocab`."""

    def __init__(self, vocab: Vocabulary, embedding_dim: int = 16):
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim

        self._register_lookup_buffers(vocab)

        self.embedding = nn.Embedding(vocab.size, embedding_dim, padding_idx=0)

        conv_in = 12 + embedding_dim
        self.convs = nn.Sequential(
            DilatedBlock(conv_in, 64, kernel_size=3, dilation=1),
            DilatedBlock(64, 64, kernel_size=3, dilation=2),
            DilatedBlock(64, 128, kernel_size=3, dilation=4),
            DilatedBlock(128, 128, kernel_size=3, dilation=8),
            DilatedBlock(128, 128, kernel_size=3, dilation=16),
        )

        # Termini live in the token sequence, so the head sees just pooled
        # conv channels + the 5 global features.
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 + 5, 128),
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
        self, new_tokens: Iterable[tuple[str, Optional[str], bool]],
    ) -> None:
        """Grow the vocab + every dependent embedding table / lookup buffer
        in-place. After calling this, any optimizer holding references to the
        old embedding parameters MUST be rebuilt.

        new_tokens: iterable of (token, parent | None, fresh: bool). The
        parent's embedding row (a residue, or <nterm>/<cterm> for a terminal
        mod) is copied for each new row (zero when there is no parent or the
        token is fresh); physico and global lookup rows are recomputed from
        vocab lineage.
        """
        new = list(new_tokens)
        if not new:
            return
        device = self.embedding.weight.device
        init = torch.zeros(len(new), self.embedding.embedding_dim, device=device)
        for k, (_, parent_aa, fresh) in enumerate(new):
            if parent_aa is None or fresh:
                continue
            pid = self.vocab.tokens.get(parent_aa)
            if pid is not None:
                init[k] = self.embedding.weight.data[pid].clone()
        for token, parent_aa, fresh in new:
            self.vocab.extend(token, parent_aa, fresh)
        self.embedding = _resize_embedding(self.embedding, self.vocab.size, init, padding_idx=0)
        self._refresh_lookup_buffers()

    # ---- forward ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x
        mask = (x != 0).unsqueeze(1)
        token_mask = (tokens != 0).float()

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
            z, gravy, net_charge, frac_aromatic,
            norm_seq_len, frac_hydrophobic,
        ], dim=1)
        return self.head(feats).squeeze(1)
