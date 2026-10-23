"""Ion-mobility prediction: the RT backbone with precursor charge as an
input channel.

Charge placement was benchmarked (results/q21_im_charge_s0.json, reinj 1/K0,
104k peptide-charge pairs): as an extra per-position conv channel it reaches
2.62% of span vs 3.67% as a head feature and 8.0% without charge — a head
scalar cannot model the charge x sequence interaction that dominates higher
charge states (z=4 MAE: 0.020 input vs 0.090 head).

Everything else mirrors the RT stack: unified-termini tokenization, robust-z
targets, mixture loss, Muon+AdamW, cosine + warmup, val-MAE selection.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from jrif.api import (
    FINETUNE_LR,
    SCRATCH_LR,
    SHIPPING_RECIPE,
    DataFrameLike,
    FitResult,
    _auto_device,
    _make_optimizers,
    _seed_everything,
)
from jrif.dataset import MAX_LEN, Vocabulary, discover_tokens, fit_rt_scale, to_polars, tokenize
from jrif.loss import LaplaceUniformMixture
from jrif.model import RTModel, DilatedBlock


class IMModel(RTModel):
    """Unified-termini RT backbone + charge as an input channel.

    forward(x, charge): charge is float (z - 2), broadcast to every position
    and concatenated with the physico + embedding channels.
    """

    def __init__(self, vocab: Vocabulary, embedding_dim: int = 16):
        super().__init__(vocab, embedding_dim=embedding_dim, unified_termini=True)
        conv_in = 12 + embedding_dim + 1
        chans, dils = [64, 64, 128, 128, 128], [1, 2, 4, 8, 16]
        blocks, prev = [], conv_in
        for c, d in zip(chans, dils):
            blocks.append(DilatedBlock(prev, c, kernel_size=3, dilation=d))
            prev = c
        self.convs = nn.Sequential(*blocks)
        head_in = prev + 3 + 1 + 1
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(head_in, 128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        tokens = x
        mask = (x != 0).unsqueeze(1)
        tm = (tokens != 0).float()
        phys = self.physico_features[x]
        emb = self.embedding(x)
        z = torch.cat([phys, emb,
                       charge.view(-1, 1, 1).expand(-1, x.size(1), 1)], dim=2)
        z = self.convs(z.permute(0, 2, 1))
        z = (z * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        L = tm.sum(dim=1, keepdim=True).clamp(min=1)
        feats = torch.cat([
            z,
            (self.hydro_table[tokens] * tm).sum(1, keepdim=True) / L,
            (self.charge_table[tokens] * tm).sum(1, keepdim=True),
            (self.aromatic_table[tokens] * tm).sum(1, keepdim=True) / L,
            L / float(MAX_LEN),
            (self.hydrophobic_table[tokens] * tm).sum(1, keepdim=True) / L,
        ], dim=1)
        return self.head(feats).squeeze(1)


class IMDataset(Dataset):
    """Tokenized peptides + centered charge + z-scored IM targets."""

    def __init__(self, df, vocab: Vocabulary, *, seq_col="seq", charge_col="charge",
                 im_col: Optional[str] = "im", im_median=None, im_mad=None):
        df = to_polars(df)
        seqs = df[seq_col].to_list()
        n = len(seqs)
        tok = np.zeros((n, MAX_LEN), dtype=np.int64)
        for i, s in enumerate(seqs):
            t, _, _ = tokenize(s, vocab, terminal_in_seq=True)
            tok[i, :] = t
        self.sequences = torch.from_numpy(tok)
        self.charges = torch.from_numpy(
            df[charge_col].to_numpy().astype(np.float32) - 2.0)
        if im_col is None:
            self.im_median = float(im_median if im_median is not None else 0.0)
            self.im_mad = float(im_mad if im_mad is not None else 1.0)
            self.targets = torch.zeros(n, dtype=torch.float32)
        else:
            ims = df[im_col].to_numpy().astype("float32")
            if im_median is None or im_mad is None:
                im_median, im_mad = fit_rt_scale(ims)
            self.im_median, self.im_mad = float(im_median), float(im_mad)
            self.targets = torch.from_numpy((ims - self.im_median) / self.im_mad)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.sequences[i], self.charges[i], self.targets[i]


class IMPredictor:
    """DataFrame-driven IM predictor: fit / predict / save / from_checkpoint."""

    def __init__(self, model: IMModel, im_median: float = 0.0, im_mad: float = 1.0):
        self.model = model
        self.im_median = float(im_median)
        self.im_mad = float(im_mad)

    @property
    def vocab(self) -> Vocabulary:
        return self.model.vocab

    @classmethod
    def from_blank(cls, *, embedding_dim: int = 16) -> "IMPredictor":
        return cls(IMModel(Vocabulary.default(), embedding_dim))

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path]) -> "IMPredictor":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        if ckpt.get("kind") != "im":
            raise ValueError(f"{path} is not an IM checkpoint")
        vocab = Vocabulary.from_dict(ckpt["vocab"])
        model = IMModel(vocab, int(ckpt.get("embedding_dim", 16)))
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, ckpt.get("im_median", 0.0), ckpt.get("im_mad", 1.0))

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 2, "kind": "im",
            "model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "vocab": self.vocab.to_dict(),
            "embedding_dim": self.model.embedding_dim,
            "im_median": self.im_median, "im_mad": self.im_mad,
        }, path)

    def predict(self, df: DataFrameLike, *, seq_col="seq", charge_col="charge",
                batch_size: int = 2048, device: Optional[str] = None) -> np.ndarray:
        device_t = torch.device(device if device else _auto_device())
        self.model.to(device_t).eval()
        ds = IMDataset(to_polars(df), self.vocab, seq_col=seq_col, charge_col=charge_col,
                       im_col=None, im_median=self.im_median, im_mad=self.im_mad)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.inference_mode():
            for tok, ch, _ in loader:
                preds.append(self.model(tok.to(device_t), ch.to(device_t)).cpu())
        z = torch.cat(preds) if preds else torch.empty(0)
        return (z * self.im_mad + self.im_median).numpy()

    def fit(self, df: DataFrameLike, *, seq_col="seq", charge_col="charge", im_col="im",
            val_df: Optional[DataFrameLike] = None, val_fraction: float = 0.1,
            epochs: int = SHIPPING_RECIPE["epochs"], patience=SHIPPING_RECIPE["patience"],
            lr: float = SCRATCH_LR, batch_size: int = SHIPPING_RECIPE["batch_size"],
            warmup_epochs: int = SHIPPING_RECIPE["warmup_epochs"],
            weight_decay: float = SHIPPING_RECIPE["weight_decay"],
            seed: int = 42, device: Optional[str] = None, verbose: bool = True) -> FitResult:
        _seed_everything(seed)
        device_t = torch.device(device if device else _auto_device())
        df_pl = (to_polars(df)
                 .select([pl.col(seq_col).alias("seq"),
                          pl.col(charge_col).cast(pl.Float64).alias("charge"),
                          pl.col(im_col).cast(pl.Float64).alias("im")])
                 .drop_nulls())
        if val_df is not None:
            val_pl = (to_polars(val_df)
                      .select([pl.col(seq_col).alias("seq"),
                               pl.col(charge_col).cast(pl.Float64).alias("charge"),
                               pl.col(im_col).cast(pl.Float64).alias("im")])
                      .drop_nulls())
            train_pl = df_pl
        else:
            seqs = df_pl["seq"].unique().to_list()
            rng = random.Random(seed); rng.shuffle(seqs)
            val_seqs = set(seqs[:max(1, int(len(seqs) * val_fraction))])
            is_val = df_pl["seq"].is_in(list(val_seqs))
            train_pl, val_pl = df_pl.filter(~is_val), df_pl.filter(is_val)

        new_main, _ = discover_tokens(train_pl["seq"].to_list() + val_pl["seq"].to_list(),
                                      self.vocab, terminal_in_seq=True)
        if new_main:
            self.model.extend_vocab(
                new_main_tokens=[(t, p, p is None) for t, p in new_main.items()])

        med, mad = fit_rt_scale(train_pl["im"].to_numpy().astype("float32"))
        self.im_median, self.im_mad = med, mad
        tr = IMDataset(train_pl, self.vocab, im_median=med, im_mad=mad)
        va = IMDataset(val_pl, self.vocab, im_median=med, im_mad=mad)
        tl = DataLoader(tr, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
        vl = DataLoader(va, batch_size=batch_size * 4, shuffle=False)

        self.model.to(device_t)
        crit = LaplaceUniformMixture(init_b=1.0, init_eps=0.05, W=10.0).to(device_t)
        opts, scheds = _make_optimizers(self.model, lr, weight_decay, epochs)
        lp = [p for p in crit.parameters() if p.requires_grad]
        if lp:
            opts[0].add_param_group({"params": lp})

        best_mae, best_state, best_epoch, wait = float("inf"), None, -1, 0
        for epoch in range(epochs):
            self.model.train()
            wf = min(1.0, (epoch + 1) / max(1, warmup_epochs))
            for o in opts:
                for g in o.param_groups:
                    g.setdefault("_base_lr", g["lr"])
                    if epoch < warmup_epochs:
                        g["lr"] = g["_base_lr"] * wf
            for tok, ch, y in tl:
                loss = crit(self.model(tok.to(device_t), ch.to(device_t)), y.to(device_t))
                loss.backward()
                for o in opts:
                    o.step(); o.zero_grad()
            if epoch >= warmup_epochs:
                for sc in scheds:
                    sc.step()
            self.model.eval()
            errs = []
            with torch.inference_mode():
                for tok, ch, y in vl:
                    errs.append((self.model(tok.to(device_t), ch.to(device_t)).cpu() - y).abs())
            vm = float(torch.cat(errs).mean())
            if verbose:
                print(f"Epoch {epoch+1:3d}/{epochs} | val_MAE {vm:.4f}", flush=True)
            if vm < best_mae:
                best_mae, best_epoch, wait = vm, epoch, 0
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in self.model.state_dict().items()})
            else:
                wait += 1
                if patience is not None and wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}", flush=True)
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(device_t)
        return FitResult(best_val_r2=float("nan"), best_val_mae=best_mae, best_epoch=best_epoch)
