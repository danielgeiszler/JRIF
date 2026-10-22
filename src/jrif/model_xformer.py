"""Dilated CNN backbone with residual skips + transformer encoder.

Architecture:
  embedding (12 physico + 16 learned = 28 ch)
    -> 5 ResDilatedBlocks (residual skip with 1x1 projection on channel change)
       channels 28 -> 64 -> 64 -> 128 -> 128 -> 128, dilations 1,2,4,8,16
    -> permute to (B, L, 128); prepend a learnable [CLS] token at position 0
    -> add sinusoidal positional encoding (length L+1)
    -> N x TransformerEncoderLayer (d_model=128, batch_first, padding-aware)
    -> take CLS output (B, 128) as the pooled sequence representation
  Head input concatenates: CLS (128) + n/c-term emb (4+4) + n/c mod (4+4)
                          + GRAVY/charge/aromatic (3) + seq_len/hydrophobic (2)
                        = 149 dims (same as RTModel)
  Head: Dropout(0.3) -> Linear(149,128) -> GELU -> Dropout(0.2) -> Linear(128,1)

CLS pooling is preferred over masked GAP after a transformer: GAP averages
attended outputs, throwing away most of the per-position structure the
attention layer just produced; CLS gives the transformer a scratchpad
position whose final-layer state is an explicit learned summary.

Token-vocab handling and mod ids match RTModel exactly so the model is a
drop-in replacement in benches and RTPredictor's training loop.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn

from jrif.dataset import Vocabulary, MAX_LEN, CANONICAL_AAS
from jrif.model import _build_physico_matrix, _build_global_tables, _resize_embedding, fast_conv1d, fast_conv_bn1d


def _sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding, shape (seq_len, d_model)."""
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                    (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ResDilatedBlock(nn.Module):
    """DilatedConv + BN + GELU with a residual skip. 1x1 conv projection on
    the skip path when in_channels != out_channels, otherwise identity."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.proj = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.act(self.bn(fast_conv1d(x, self.conv))) + self.proj(x)
        return self.act(fast_conv_bn1d(x, self.conv, self.bn)) + self.proj(x)


class XformerSkipModel(nn.Module):
    """Dilated CNN with residual skips + transformer encoder + masked GAP head."""

    def __init__(
        self,
        vocab: Vocabulary,
        embedding_dim: int = 16,
        d_model: int = 128,
        nhead: int = 4,
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
        # PE length is MAX_LEN + 1 to cover the prepended CLS at position 0.
        self.register_buffer("pos_enc", _sinusoidal_pe(MAX_LEN + 1, d_model), persistent=False)

        # Learnable CLS token (B, 1, d_model after expansion). Init small.
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        head_in = d_model + 4 + 4 + 4 + 4 + 3 + 1 + 1
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def _refresh_lookup_buffers(self) -> None:
        device = self.physico_features.device
        self.physico_features = _build_physico_matrix(self.vocab).to(device)
        h, c, a, hd = _build_global_tables(self.vocab)
        self.hydro_table = h.to(device)
        self.charge_table = c.to(device)
        self.aromatic_table = a.to(device)
        self.hydrophobic_table = hd.to(device)

    def extend_vocab(
        self,
        new_main_tokens: Iterable[tuple[str, Optional[str], bool]] = (),
        new_nterm_mods: Iterable[str] = (),
        new_cterm_mods: Iterable[str] = (),
    ) -> None:
        """Grow the vocab + every dependent embedding table / lookup buffer
        in-place. After calling this, any optimizer holding references to the
        old embedding parameters MUST be rebuilt."""
        new_main = list(new_main_tokens)
        new_nterm = list(new_nterm_mods)
        new_cterm = list(new_cterm_mods)
        if not new_main and not new_nterm and not new_cterm:
            return

        device = self.embedding.weight.device

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

    def forward(
        self,
        x: torch.Tensor,
        nterm_mod_id: Optional[torch.Tensor] = None,
        cterm_mod_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.size(0)
        tokens = x
        token_mask = (tokens != 0).float()           # (B, L)

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

        phys = self.physico_features[x]              # (B, L, 12)
        emb = self.embedding(x)                      # (B, L, d_emb)
        z = torch.cat([phys, emb], dim=2).permute(0, 2, 1)   # (B, 28, L)
        z = self.convs(z)                            # (B, d_model, L)

        # Prepend CLS, build padding mask (CLS is never padded).
        L = x.size(1)
        z_t = z.permute(0, 2, 1)                                          # (B, L, d)
        cls = self.cls_token.expand(B, -1, -1)                            # (B, 1, d)
        z_t = torch.cat([cls, z_t], dim=1)                                # (B, L+1, d)
        z_t = z_t + self.pos_enc[: L + 1]                                 # PE incl. CLS
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
        pad_mask = torch.cat([cls_pad, x == 0], dim=1)                    # (B, L+1)
        z_t = self.transformer(z_t, src_key_padding_mask=pad_mask)        # (B, L+1, d)
        z = z_t[:, 0, :]                                                  # CLS output (B, d)

        seq_lengths = token_mask.sum(dim=1, keepdim=True).clamp(min=1)
        gravy           = (self.hydro_table[tokens]       * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        net_charge      = (self.charge_table[tokens]      * token_mask).sum(dim=1, keepdim=True)
        frac_aromatic   = (self.aromatic_table[tokens]    * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        frac_hydrophobic= (self.hydrophobic_table[tokens] * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        norm_seq_len    = seq_lengths / float(MAX_LEN)

        feats = torch.cat([
            z, n_term_emb, c_term_emb, n_mod, c_mod,
            gravy, net_charge, frac_aromatic, norm_seq_len, frac_hydrophobic,
        ], dim=1)
        return self.head(feats).squeeze(1)
