"""Dilated-CNN retention-time model.

Architecture overview:
  Token indices -> [physicochemical features (12) + learned embedding (16) = 28 ch]
    -> 5 dilated Conv1d blocks (k=3; dilations 1,2,4,8,16; GELU + BN)
    -> masked GAP -> [128 ch]
  Head input concatenates:
    - GAP output (128)
    - n_term_embedding(n_term_token)  (4)
    - c_term_embedding(c_term_token)  (4)
    - nterm_mod_embedding(nterm_mod_id)  (4)
    - cterm_mod_embedding(cterm_mod_id)  (4)
    - GRAVY (1), net charge @ pH 2.7 (1), frac aromatic (1)
    - normalized seq length (1), frac hydrophobic (1)
    = 149 dims
  FC head: Dropout(0.3) -> Linear(149, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)
"""

import torch
import torch.nn as nn

from jrif.dataset import VOCAB_SIZE, MAX_LEN, NTERM_MOD_VOCAB_SIZE, CTERM_MOD_VOCAB_SIZE


def load_state_dict_compat(model: nn.Module, state_dict: dict) -> None:
    """Best-effort state-dict load: copy matching-shape tensors, skip mismatches."""
    new = model.state_dict()
    for key in list(new.keys()):
        if key in state_dict and new[key].shape == state_dict[key].shape:
            new[key] = state_dict[key]
    model.load_state_dict(new)


def _build_physico_matrix() -> torch.Tensor:
    """Per-residue physicochemical features, z-scored across the 20 canonical AAs.
    Rows: [pad, A..Y (canonical order), M[147], C[160], K[308]]. Shape [24, 12].
    Features (order): hydropathy (Kyte-Doolittle), mass, side-chain pKa,
    vdW volume, aromaticity, H-bond donors, H-bond acceptors, pI, helix
    propensity, sheet propensity, is-Lys flag, is-Arg flag."""
    raw = torch.tensor([
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
    ], dtype=torch.float32)
    mean = raw.mean(dim=1, keepdim=True)
    std = raw.std(dim=1, keepdim=True).clamp(min=1e-6)
    raw = (raw - mean) / std
    matrix = torch.zeros(VOCAB_SIZE, 12)
    matrix[1:21] = raw.T                    # canonical AAs (indices 1-20)
    matrix[21] = matrix[11]                 # M[147] <- M (backbone chemistry preserved)
    matrix[22] = matrix[2]                  # C[160] <- C
    # K[308] row stays zero: the 308 Da side-chain tag changes chemistry enough
    # that we don't want to inherit K's features. Main embedding learns it.
    return matrix


def _build_global_feature_tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token lookup tables for whole-peptide scalar features."""
    kd = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]
    charge = [0, 0, 0.9, 0.97, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0]
    aromatic = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    hydrophobic = [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0]

    hydro = torch.zeros(VOCAB_SIZE)
    chg = torch.zeros(VOCAB_SIZE)
    aro = torch.zeros(VOCAB_SIZE)
    hyd = torch.zeros(VOCAB_SIZE)
    for i in range(20):
        hydro[i + 1] = kd[i]
        chg[i + 1] = charge[i]
        aro[i + 1] = aromatic[i]
        hyd[i + 1] = hydrophobic[i]
    hydro[21] = hydro[11]; hydro[22] = hydro[2]
    chg[21] = chg[11]; chg[22] = chg[2]
    aro[21] = aro[11]; aro[22] = aro[2]
    hyd[21] = hyd[11]; hyd[22] = hyd[2]
    # K[308] keeps zero across all four (tag chemistry, not known a priori)
    return hydro, chg, aro, hyd


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
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        # Deterministic lookup buffers (non-persistent; recomputed at construction).
        self.register_buffer("physico_features", _build_physico_matrix(), persistent=False)
        hydro, chg, aro, hyd = _build_global_feature_tables()
        self.register_buffer("hydro_table", hydro, persistent=False)
        self.register_buffer("charge_table", chg, persistent=False)
        self.register_buffer("aromatic_table", aro, persistent=False)
        self.register_buffer("hydrophobic_table", hyd, persistent=False)

        self.embedding = nn.Embedding(VOCAB_SIZE, 16, padding_idx=0)
        self.n_term_embedding = nn.Embedding(VOCAB_SIZE, 4, padding_idx=0)
        self.c_term_embedding = nn.Embedding(VOCAB_SIZE, 4, padding_idx=0)
        self.nterm_mod_embedding = nn.Embedding(NTERM_MOD_VOCAB_SIZE, 4)
        self.cterm_mod_embedding = nn.Embedding(CTERM_MOD_VOCAB_SIZE, 4)

        self.convs = nn.Sequential(
            DilatedBlock(28, 64, kernel_size=3, dilation=1),
            DilatedBlock(64, 64, kernel_size=3, dilation=2),
            DilatedBlock(64, 128, kernel_size=3, dilation=4),
            DilatedBlock(128, 128, kernel_size=3, dilation=8),
            DilatedBlock(128, 128, kernel_size=3, dilation=16),
        )

        # Head input: 128 (GAP) + 4+4 (terminal AA) + 4+4 (terminal mods)
        #           + 3 (gravy/charge/aromatic) + 1+1 (seq_len/hydrophobic)
        #           = 149
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(149, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        nterm_mod_id: torch.Tensor = None,
        cterm_mod_id: torch.Tensor = None,
    ) -> torch.Tensor:
        B = x.size(0)
        tokens = x
        mask = (x != 0).unsqueeze(1)                            # [B, 1, L]
        token_mask = (tokens != 0).float()                      # [B, L]

        # Terminal AA identity (raw token; K[308] at terminus gets its own row)
        n_term_token = tokens[:, 0]
        lengths = (x != 0).sum(dim=1)
        c_term_idx = (lengths - 1).clamp(min=0)
        c_term_tokens = x.gather(1, c_term_idx.unsqueeze(1)).squeeze(1)
        n_term_emb = self.n_term_embedding(n_term_token)
        c_term_emb = self.c_term_embedding(c_term_tokens)

        # Terminal mods
        if nterm_mod_id is None:
            nterm_mod_id = torch.zeros(B, dtype=torch.long, device=tokens.device)
        if cterm_mod_id is None:
            cterm_mod_id = torch.zeros(B, dtype=torch.long, device=tokens.device)
        n_mod = self.nterm_mod_embedding(nterm_mod_id)
        c_mod = self.cterm_mod_embedding(cterm_mod_id)

        # Sequence path
        phys = self.physico_features[x]                         # [B, L, 12]
        emb = self.embedding(x)                                 # [B, L, 16]
        z = torch.cat([phys, emb], dim=2).permute(0, 2, 1)      # [B, 28, L]
        z = self.convs(z)                                       # [B, 128, L]
        z = (z * mask).sum(dim=2) / mask.sum(dim=2)             # masked GAP -> [B, 128]

        # Global scalar features
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
