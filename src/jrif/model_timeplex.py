"""timePlex CNN architecture (ParallelSquared) ported to PyTorch / JRIF.

Faithful reproduction of CNN_LF_iRT.ipynb:
  - One-hot input (no learned embedding)
  - 4× Conv1d + MaxPool with SiLU activation
  - Large FC head (1000→256→128→64→32→1)
  - No batch norm, no dropout
  - Adapted to use JRIF's tokenization (variable vocab, MAX_LEN=50) instead of
    the original (20-AA one-hot, MAX_LEN=30).

Forward signature is (seqs, nterm_mod_id, cterm_mod_id) for compatibility with
RTPredictor's training loop, but the last two arguments are ignored. timePlex
can't represent peptide-level mods.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jrif.dataset import MAX_LEN, Vocabulary


class TimePlexModel(nn.Module):
    """timePlex CNN architecture adapted for JRIF's tokenization."""

    def __init__(self, vocab: Vocabulary):
        super().__init__()
        self.vocab = vocab
        v = vocab.size
        self.conv1 = nn.Conv1d(v, 32, kernel_size=4, padding="same")
        self.pool1 = nn.MaxPool1d(kernel_size=1)  # no-op, matches their code
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, padding="same")
        self.pool2 = nn.MaxPool1d(kernel_size=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, padding="same")
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=4, padding="same")
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        flat_dim = 256 * (MAX_LEN // 4)
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 1000), nn.SiLU(),
            nn.Linear(1000, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1),
        )

    @property
    def embedding_dim(self) -> int:
        # No learned embedding — exposed for symmetry with RTModel's interface.
        return 0

    def forward(
        self,
        x: torch.Tensor,
        nterm_mod_id: torch.Tensor | None = None,  # ignored
        cterm_mod_id: torch.Tensor | None = None,  # ignored
    ) -> torch.Tensor:
        x = F.one_hot(x, num_classes=self.vocab.size).float()
        x = x.permute(0, 2, 1)
        x = self.pool1(F.silu(self.conv1(x)))
        x = self.pool2(F.silu(self.conv2(x)))
        x = self.pool3(F.silu(self.conv3(x)))
        x = self.pool4(F.silu(self.conv4(x)))
        x = x.flatten(1)
        return self.fc(x).squeeze(1)
