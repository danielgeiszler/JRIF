"""Pure transformer (no CNN backbone) retention-time model.

Architecture:
  embedding (12 physico + 16 learned = 28 ch)
    -> Linear(28, d_model) input projection
    -> prepend learnable [CLS]; add sinusoidal PE (length L+1)
    -> N x TransformerEncoderLayer (norm_first, batch_first, padding-aware)
    -> CLS output (B, d_model) as the pooled sequence representation
  Head input concatenates: CLS (d_model) + n/c-term emb (4+4) + n/c mod (4+4)
                          + GRAVY/charge/aromatic (3) + seq_len/hydrophobic (2)
                        = d_model + 21 dims
  Head: Dropout(0.3) -> Linear(d_model+21, 128) -> GELU -> Dropout(0.2) -> Linear(128,1)

This is the smallest pure-transformer variant in the family — replaces the
5 ResDilatedBlocks in XformerSkipModel with a single 28->d_model linear
projection (~150k fewer params at d_model=128).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from jrif.dataset import Vocabulary, MAX_LEN
from jrif.model import _build_physico_matrix, _build_global_tables
from jrif.model_xformer import _sinusoidal_pe


class XformerPureModel(nn.Module):
    """Pure transformer encoder (no CNN). CLS-token pooled."""

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
        self.register_buffer("pos_enc", _sinusoidal_pe(MAX_LEN + 1, d_model), persistent=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.embedding = nn.Embedding(vocab.size, embedding_dim, padding_idx=0)
        self.n_term_embedding = nn.Embedding(vocab.size, 4, padding_idx=0)
        self.c_term_embedding = nn.Embedding(vocab.size, 4, padding_idx=0)
        self.nterm_mod_embedding = nn.Embedding(max(vocab.nterm_size, 1), 4)
        self.cterm_mod_embedding = nn.Embedding(max(vocab.cterm_size, 1), 4)

        in_ch = 12 + embedding_dim
        self.input_proj = nn.Linear(in_ch, d_model)

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

    def forward(
        self,
        x: torch.Tensor,
        nterm_mod_id: Optional[torch.Tensor] = None,
        cterm_mod_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        phys = self.physico_features[x]                  # (B, L, 12)
        emb = self.embedding(x)                          # (B, L, d_emb)
        z = torch.cat([phys, emb], dim=2)                # (B, L, 28)
        z = self.input_proj(z)                           # (B, L, d_model)

        L = x.size(1)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, d_model)
        z = torch.cat([cls, z], dim=1)                   # (B, L+1, d_model)
        z = z + self.pos_enc[: L + 1]
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
        pad_mask = torch.cat([cls_pad, x == 0], dim=1)
        z = self.transformer(z, src_key_padding_mask=pad_mask)
        z = z[:, 0, :]                                   # CLS output (B, d_model)

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
