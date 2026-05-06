"""Variant of XformerSkipModel using Interleaved Head Attention (IHA) in
place of standard multi-head attention.

Reference: Duvvuri et al., "Interleaved Head Attention," arXiv:2602.21371.
The implementation follows Algorithm 1 of the paper.

IHA replaces MHA's Q, K, V with cross-head mixed pseudo-heads:
  Q'[b, h, p, n, :] = sum_m Theta_Q[m, h, p] * Q[b, n, m, :]
(and similarly for K', V'), then runs standard scaled-dot-product attention
on a length-(N*P) interleaved sequence per base head, then collapses HP
output channels back to H via a learned R in R^(H, H*P).

Theorem 2 of the paper shows MHA is a special case of IHA. We initialize
Theta_* and R so that the layer starts close to a standard H-head MHA.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from jrif.dataset import Vocabulary, MAX_LEN
from jrif.model import _build_physico_matrix, _build_global_tables
from jrif.model_xformer import ResDilatedBlock, _sinusoidal_pe


class InterleavedHeadAttention(nn.Module):
    """Self-attention layer implementing IHA (Algorithm 1 of arXiv:2602.21371).

    Drop-in replacement for nn.MultiheadAttention(batch_first=True), supporting
    `key_padding_mask` of shape (B, N) where True marks pad positions.
    """

    def __init__(self, d_model: int, num_heads: int, num_pseudo: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        if num_pseudo is None:
            num_pseudo = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.H = num_heads
        self.P = num_pseudo
        self.d = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Pseudo-head mixing tensors. Indexing convention: theta[m, h, p] mixes
        # original head m into pseudo (h, p).
        self.theta_Q = nn.Parameter(torch.empty(self.H, self.H, self.P))
        self.theta_K = nn.Parameter(torch.empty(self.H, self.H, self.P))
        self.theta_V = nn.Parameter(torch.empty(self.H, self.H, self.P))

        # Output collapse map: HP-many pseudo outputs -> H output heads.
        self.R = nn.Parameter(torch.empty(self.H, self.H * self.P))

        self.attn_dropout = nn.Dropout(dropout)
        self._reset_iha_params()

    def _reset_iha_params(self) -> None:
        """Init Theta / R close to the MHA-equivalent assignment so the layer
        starts as a standard H-head attention and gradually learns to use the
        pseudo-head mixing."""
        with torch.no_grad():
            # Theta: identity-on-j=0, small Gaussian noise elsewhere.
            for theta in (self.theta_Q, self.theta_K, self.theta_V):
                theta.zero_()
                for m in range(self.H):
                    theta[m, m, 0] = 1.0
                theta.add_(0.02 * torch.randn_like(theta))
            # R: select pseudo-(h, 0) for output head h, small noise elsewhere.
            self.R.zero_()
            for h in range(self.H):
                self.R[h, h * self.P + 0] = 1.0
            self.R.add_(0.02 * torch.randn_like(self.R))

    def forward(
        self, x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, N, D). key_padding_mask: (B, N) bool, True = padded.
        Returns: (B, N, D)."""
        B, N, D = x.shape
        H, P, d = self.H, self.P, self.d

        Q = self.W_Q(x).view(B, N, H, d)
        K = self.W_K(x).view(B, N, H, d)
        V = self.W_V(x).view(B, N, H, d)

        # Cross-head mixing: (B, N, H, d), Theta (H_m, H_h, P) -> (B, H, P, N, d).
        # einsum: theta[m, h, p] * Q[b, n, m, d] -> Qp[b, h, p, n, d]
        Qp = torch.einsum("mhp,bnmd->bhpnd", self.theta_Q, Q)
        Kp = torch.einsum("mhp,bnmd->bhpnd", self.theta_K, K)
        Vp = torch.einsum("mhp,bnmd->bhpnd", self.theta_V, V)

        # Interleaved merge: (B, H, P, N, d) -> (B, H, N*P, d) with order
        # (n=0, p=0), (n=0, p=1), ..., (n=0, p=P-1), (n=1, p=0), ...
        # i.e., permute to (B, H, N, P, d) then reshape to (B, H, N*P, d).
        Qp = Qp.permute(0, 1, 3, 2, 4).reshape(B, H, N * P, d)
        Kp = Kp.permute(0, 1, 3, 2, 4).reshape(B, H, N * P, d)
        Vp = Vp.permute(0, 1, 3, 2, 4).reshape(B, H, N * P, d)

        # Standard scaled dot-product attention per head on length-(N*P).
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(Qp, Kp.transpose(-2, -1)) * scale     # (B, H, NP, NP)

        if key_padding_mask is not None:
            # Expand pad mask: each padded original token contributes P
            # consecutive padded pseudo-positions in the merged sequence.
            pad_np = key_padding_mask.unsqueeze(-1).expand(B, N, P).reshape(B, N * P)
            # Mask keys (last dim of scores) — broadcast over query dim.
            scores = scores.masked_fill(
                pad_np.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, NP)
                float("-inf"),
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, Vp)                                # (B, H, NP, d)

        # Unmerge: (B, H, NP, d) -> (B, H, N, P, d) consistent with the
        # (n-major, p-minor) interleave we just performed.
        out = out.view(B, H, N, P, d)

        # Collapse HP -> H via R. Flatten over (h', j) so the column index
        # matches R's HP dimension. Use the SAME flatten order R was built
        # against: column = h' * P + j.
        out_flat = out.permute(0, 1, 3, 2, 4).reshape(B, H * P, N, d)
        # einsum: R[h, o] * out_flat[b, o, n, d] -> O[b, h, n, d]
        out = torch.einsum("ho,bond->bhnd", self.R, out_flat)        # (B, H, N, d)

        # Concat heads and output project.
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.W_O(out)


class IHAEncoderLayer(nn.Module):
    """A single transformer encoder layer with IHA self-attention. norm_first
    layout (pre-LN), to match the XformerSkipModel reference encoder."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 512,
        dropout: float = 0.1, num_pseudo: Optional[int] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = InterleavedHeadAttention(d_model, nhead, num_pseudo=num_pseudo, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # Pre-LN
        x = x + self.dropout1(self.attn(self.norm1(x), key_padding_mask=src_key_padding_mask))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class IHAEncoder(nn.Module):
    def __init__(self, layer: IHAEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer if i == 0 else IHAEncoderLayer(
            d_model=layer.norm1.normalized_shape[0],
            nhead=layer.attn.H,
            dim_feedforward=layer.ff[0].out_features,
            dropout=layer.dropout1.p,
            num_pseudo=layer.attn.P,
        ) for i in range(num_layers)])

    def forward(self, x, src_key_padding_mask=None):
        for lyr in self.layers:
            x = lyr(x, src_key_padding_mask=src_key_padding_mask)
        return x


class XformerSkipIHAModel(nn.Module):
    """Identical to XformerSkipModel but the transformer uses IHA in place of
    standard multi-head attention."""

    def __init__(
        self,
        vocab: Vocabulary,
        embedding_dim: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_pseudo: Optional[int] = None,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.d_model = d_model

        self.register_buffer("physico_features", _build_physico_matrix(vocab), persistent=False)
        h, c, a, hd = _build_global_tables(vocab)
        self.register_buffer("hydro_table", h, persistent=False)
        self.register_buffer("charge_table", c, persistent=False)
        self.register_buffer("aromatic_table", a, persistent=False)
        self.register_buffer("hydrophobic_table", hd, persistent=False)
        self.register_buffer("pos_enc", _sinusoidal_pe(MAX_LEN + 1, d_model), persistent=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.embedding = nn.Embedding(vocab.size, embedding_dim, padding_idx=0)
        self.n_term_embedding = nn.Embedding(vocab.size, 4, padding_idx=0)
        self.c_term_embedding = nn.Embedding(vocab.size, 4, padding_idx=0)
        self.nterm_mod_embedding = nn.Embedding(max(vocab.nterm_size, 1), 4)
        self.cterm_mod_embedding = nn.Embedding(max(vocab.cterm_size, 1), 4)

        conv_in = 12 + embedding_dim
        self.convs = nn.Sequential(
            ResDilatedBlock(conv_in, 64, kernel_size=3, dilation=1),
            ResDilatedBlock(64, 64, kernel_size=3, dilation=2),
            ResDilatedBlock(64, 128, kernel_size=3, dilation=4),
            ResDilatedBlock(128, 128, kernel_size=3, dilation=8),
            ResDilatedBlock(128, d_model, kernel_size=3, dilation=16),
        )

        first = IHAEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, num_pseudo=num_pseudo,
        )
        self.transformer = IHAEncoder(first, num_layers=num_layers)

        head_in = d_model + 4 + 4 + 4 + 4 + 3 + 1 + 1
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x, nterm_mod_id=None, cterm_mod_id=None):
        B = x.size(0)
        tokens = x
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

        L = x.size(1)
        z_t = z.permute(0, 2, 1)
        cls = self.cls_token.expand(B, -1, -1)
        z_t = torch.cat([cls, z_t], dim=1)
        z_t = z_t + self.pos_enc[: L + 1]
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
        pad_mask = torch.cat([cls_pad, x == 0], dim=1)
        z_t = self.transformer(z_t, src_key_padding_mask=pad_mask)
        z = z_t[:, 0, :]

        seq_lengths = token_mask.sum(dim=1, keepdim=True).clamp(min=1)
        gravy            = (self.hydro_table[tokens]       * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        net_charge       = (self.charge_table[tokens]      * token_mask).sum(dim=1, keepdim=True)
        frac_aromatic    = (self.aromatic_table[tokens]    * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        frac_hydrophobic = (self.hydrophobic_table[tokens] * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        norm_seq_len     = seq_lengths / float(MAX_LEN)

        feats = torch.cat([
            z, n_term_emb, c_term_emb, n_mod, c_mod,
            gravy, net_charge, frac_aromatic, norm_seq_len, frac_hydrophobic,
        ], dim=1)
        return self.head(feats).squeeze(1)
