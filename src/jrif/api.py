"""Public DataFrame-driven API for JRIF.

`RTPredictor` is the single user-facing entrypoint. Build one of three ways:

  1. `RTPredictor.from_blank()`  — fresh model with the default 24-token vocab.
  2. `RTPredictor.from_checkpoint(path)` — reload a previously-trained model
     (self-describing checkpoint OR a legacy pre-format_version `.pt` file).
  3. After `fit`, the same instance carries a trained model with vocab and
     RT scale already wired up, ready for `.predict(df)` and `.save(path)`.

Calling `.fit(df)` will:
  - discover any modification tokens in the input that the model's vocab does
    not know yet,
  - extend the embedding tables (inheriting from parent AAs by default;
    `fresh_mods=[...]` opts new tokens out of inheritance),
  - robust z-score the RT column (median / MAD on the train split),
  - train (Muon for conv weights, AdamW elsewhere; cosine + linear warmup;
    optional early stopping on val R²), and
  - keep the best-val-R² state in `self.model`.

Either polars or pandas DataFrames are accepted; everything internal is polars.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jrif.dataset import (
    MAX_LEN,
    RTDataset,
    Vocabulary,
    discover_tokens,
    fit_rt_scale,
    to_polars,
)
from jrif.loss import LaplaceUniformMixture, NormalNormalMixture
from jrif.model import RTModel, load_legacy_state_dict
from jrif.model_xformer import XformerSkipModel
from jrif.muon import Muon


CHECKPOINT_FORMAT_VERSION = 1


# Architecture registry: name -> (build_fn(vocab, embedding_dim), arch_label).
# Only architectures listed here can be saved to / loaded from a checkpoint.
def _build_dilated_cnn(vocab: Vocabulary, embedding_dim: int,
                       unified_termini: bool = False) -> nn.Module:
    return RTModel(vocab, embedding_dim=embedding_dim, unified_termini=unified_termini)


def _build_xformer_skip(vocab: Vocabulary, embedding_dim: int,
                        unified_termini: bool = False) -> nn.Module:
    if unified_termini:
        raise ValueError("unified_termini is not yet supported for xformer_skip")
    return XformerSkipModel(
        vocab, embedding_dim=embedding_dim,
        d_model=128, nhead=4, num_layers=2,
        dim_feedforward=512, dropout=0.1,
    )


_ARCH_BUILDERS = {
    "dilated_cnn":  _build_dilated_cnn,
    "xformer_skip": _build_xformer_skip,
}
DEFAULT_ARCH = "xformer_skip"


DataFrameLike = Union[pl.DataFrame, pd.DataFrame]

# The settled shipping recipe (bench_transfer_control / bench_termmods_pooling):
# batch 128 + lr 1e-3 from scratch beat every batch/lr variant tested; mixture
# loss + val-MAE selection handle PSM label noise. Fine-tunes use lr 1e-4.
SHIPPING_RECIPE = dict(
    epochs=150, patience=20, batch_size=128,
    warmup_epochs=3, weight_decay=1e-4,
    loss="mixture", selection_metric="mae",
)
SCRATCH_LR = 1e-3
# Fine-tune LR bracketed on reinj FT data (q8/q9): 3e-4 < 1e-3 > 3e-3; the
# old 1e-4 convention was never measured and loses ~9% MAE. (1 seed, 1 dataset.)
FINETUNE_LR = 1e-3


@dataclass
class FitResult:
    best_val_r2: float
    best_val_mae: float
    best_epoch: int
    history: list[dict] = field(default_factory=list)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _split_train_val(
    df: pl.DataFrame, val_fraction: float, seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Peptide-level split (avoids leakage when the same seq is repeated)."""
    seqs = df["seq"].unique().to_list()
    rng = random.Random(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(len(seqs) * val_fraction))
    val_seqs = set(seqs[:n_val])
    is_val = df["seq"].is_in(list(val_seqs))
    return df.filter(~is_val), df.filter(is_val)


def _make_optimizers(
    model: nn.Module, lr: float, weight_decay: float, t_max: int,
) -> tuple[list[torch.optim.Optimizer], list]:
    muon_params, adamw_params = [], []
    for name, pr in model.named_parameters():
        if not pr.requires_grad:
            continue
        if "convs." in name and "conv.weight" in name and pr.ndim >= 2:
            muon_params.append(pr)
        else:
            adamw_params.append(pr)
    opts: list[torch.optim.Optimizer] = []
    scheds = []
    if adamw_params:
        opt = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)
        opts.append(opt)
        scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_max)))
    if muon_params:
        opt = Muon(muon_params, lr=lr * 5, momentum=0.95)
        opts.append(opt)
        scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_max)))
    return opts, scheds


def _eval_metrics(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
) -> tuple[float, float, float, float]:
    """Returns (loss, mae, r2, delta95) in z-space."""
    model.eval()
    total_loss = 0.0
    preds_all, tgts_all = [], []
    with torch.no_grad():
        for seqs, nterm, cterm, targets in loader:
            seqs = seqs.to(device); nterm = nterm.to(device)
            cterm = cterm.to(device); targets = targets.to(device)
            preds = model(seqs, nterm, cterm)
            total_loss += criterion(preds, targets).item() * len(targets)
            preds_all.append(preds.cpu()); tgts_all.append(targets.cpu())
    preds = torch.cat(preds_all); tgts = torch.cat(tgts_all)
    n = len(tgts)
    err = (preds - tgts).abs()
    mae = err.mean().item()
    d95 = err.quantile(0.95).item()
    ss_res = ((tgts - preds) ** 2).sum().item()
    ss_tot = ((tgts - tgts.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return total_loss / n, mae, r2, d95


class RTPredictor:
    """DataFrame-driven RT predictor wrapping an RTModel + Vocabulary + RT scale."""

    def __init__(
        self,
        model: nn.Module,
        rt_median: float = 0.0,
        rt_mad: float = 1.0,
        trained_on: Optional[str] = None,
        arch: str = DEFAULT_ARCH,
    ):
        self.model = model
        self.rt_median = float(rt_median)
        self.rt_mad = float(rt_mad)
        self.trained_on = trained_on
        self.arch = arch

    @property
    def vocab(self) -> Vocabulary:
        return self.model.vocab

    # ---- constructors ----

    @classmethod
    def from_blank(
        cls, *, embedding_dim: int = 16, arch: str = DEFAULT_ARCH,
        unified_termini: bool = False,
    ) -> "RTPredictor":
        """unified_termini: terminal mods are main-vocab tokens in the
        sequence (format_version-2 checkpoints); no terminal head features."""
        if arch not in _ARCH_BUILDERS:
            raise ValueError(f"Unknown arch {arch!r}. Available: {sorted(_ARCH_BUILDERS)}")
        model = _ARCH_BUILDERS[arch](Vocabulary.default(), embedding_dim, unified_termini)
        return cls(model=model, arch=arch)

    @classmethod
    def from_checkpoint(
        cls, path: Union[str, Path], *, device: Optional[str] = None,
    ) -> "RTPredictor":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("format_version") in (1, 2):
            vocab = Vocabulary.from_dict(ckpt["vocab"])
            embedding_dim = int(ckpt.get("embedding_dim", 16))
            # Default to dilated_cnn for v1 checkpoints written before arch was
            # tracked, so older saved files still load without a key error.
            arch = ckpt.get("arch", "dilated_cnn")
            if arch not in _ARCH_BUILDERS:
                raise ValueError(f"Checkpoint arch {arch!r} not supported. "
                                 f"Available: {sorted(_ARCH_BUILDERS)}")
            unified = bool(ckpt.get("unified_termini", False))
            model = _ARCH_BUILDERS[arch](vocab, embedding_dim, unified)
            model.load_state_dict(ckpt["model_state_dict"])
            predictor = cls(
                model=model,
                rt_median=float(ckpt.get("rt_median", 0.0)),
                rt_mad=float(ckpt.get("rt_mad", 1.0)),
                trained_on=ckpt.get("trained_on"),
                arch=arch,
            )
        else:
            # Legacy checkpoint: build a default-vocab dilated CNN and slice
            # oversized mod embeddings down to fit.
            model = RTModel(Vocabulary.default(), embedding_dim=16)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            load_legacy_state_dict(model, state)
            rt_med = float(ckpt.get("rt_median", 0.0)) if isinstance(ckpt, dict) else 0.0
            rt_mad = float(ckpt.get("rt_mad", 1.0)) if isinstance(ckpt, dict) else 1.0
            predictor = cls(model=model, rt_median=rt_med, rt_mad=rt_mad,
                            trained_on="legacy", arch="dilated_cnn")
        if device is not None:
            predictor.model.to(device)
        return predictor

    # ---- save ----

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        unified = bool(getattr(self.model, "unified_termini", False))
        torch.save({
            "format_version": 2 if unified else CHECKPOINT_FORMAT_VERSION,
            "unified_termini": unified,
            "arch": self.arch,
            "model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "vocab": self.vocab.to_dict(),
            "embedding_dim": self.model.embedding_dim,
            "rt_median": self.rt_median,
            "rt_mad": self.rt_mad,
            "trained_on": self.trained_on,
        }, path)

    # ---- predict ----

    def predict(
        self,
        df: DataFrameLike,
        *,
        seq_col: str = "seq",
        batch_size: int = 2048,
        device: Optional[str] = None,
    ) -> np.ndarray:
        device_t = torch.device(
            device if device else _auto_device()
        )
        self.model.to(device_t)
        self.model.eval()

        df_pl = to_polars(df)
        # RTDataset wants a Retention/rt placeholder; we don't use it.
        df_with_rt = df_pl.with_columns(pl.lit(0.0).alias("__rt__"))
        ds = RTDataset(
            df_with_rt, self.vocab,
            seq_col=seq_col, rt_col="__rt__",
            rt_median=self.rt_median, rt_mad=self.rt_mad,
            terminal_in_seq=bool(getattr(self.model, "unified_termini", False)),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        preds_z: list[torch.Tensor] = []
        with torch.inference_mode():
            for seqs, nterm, cterm, _ in loader:
                z = self.model(seqs.to(device_t), nterm.to(device_t), cterm.to(device_t)).cpu()
                preds_z.append(z)
        z = torch.cat(preds_z) if preds_z else torch.empty(0)
        return (z * self.rt_mad + self.rt_median).numpy()

    # ---- fit ----

    def fit(
        self,
        df: DataFrameLike,
        *,
        seq_col: str = "seq",
        rt_col: str = "rt",
        val_df: Optional[DataFrameLike] = None,
        val_fraction: float = 0.1,
        epochs: int = 50,
        lr: float = 1e-4,
        batch_size: Union[int, str] = 128,
        warmup_epochs: int = 3,
        weight_decay: float = 1e-4,
        fresh_mods: Sequence[str] = (),
        seed: int = 42,
        device: Optional[str] = None,
        num_threads: Optional[int] = None,
        log_dir: Optional[str] = None,
        patience: Optional[int] = None,
        trained_on: Optional[str] = None,
        verbose: bool = True,
        loss: str = "mae",
        selection_metric: str = "mae",
    ) -> FitResult:
        """Train the wrapped model on a DataFrame.

        loss:
          "mae"     — nn.L1Loss; recommended for clean / curated libraries.
          "huber"   — nn.HuberLoss(delta=1.0); kept for backwards compat.
          "mixture" — LaplaceUniformMixture with learnable b, eps; recommended
                      for prefiltered-but-noisy data (PSMtag / FragPipe PSMs).
          "normal_mixture" — NormalNormalMixture (Gaussian bulk + wider
                      Gaussian outlier component, all scales learned).

        selection_metric:
          "mae"     — keep the checkpoint with the lowest val MAE (recommended;
                      R² is saturated above 0.99 and noisy in the third decimal).
          "r2"      — legacy: keep the checkpoint with the highest val R².

        num_threads:
          CPU only: torch intra-op threads. Defaults to torch's own choice
          (the performance-core count on Apple Silicon), which benchmarks as
          optimal — pass a smaller value to leave cores free for other work.

        batch_size:
          An int, or "auto" to pick a throughput-optimal batch from the
          compute budget (CPU: min(1024, 128 × threads); GPU: 1024).
          `epochs` and `patience` are denominated in epochs at the reference
          batch of 128: when batch_size exceeds it, both are rescaled by
          (batch / 128) so the total optimizer-STEP budget stays constant —
          step-matched large batches tie the batch-128 recipe (q5), while a
          fixed epoch cap silently starves them (q4). lr is NOT rescaled:
          unscaled lr beat sqrt- and linear-scaled variants at batch 1024.
        """
        _seed_everything(seed)
        device_t = torch.device(
            device if device else _auto_device()
        )
        if num_threads is not None and device_t.type == "cpu":
            torch.set_num_threads(max(1, num_threads))
        if batch_size == "auto":
            if device_t.type == "cpu":
                batch_size = min(1024, 128 * torch.get_num_threads())
            else:
                batch_size = 1024
            if verbose:
                print(f"Auto batch size: {batch_size}", flush=True)
        if batch_size > 128:
            scale = batch_size / 128
            epochs = max(1, int(round(epochs * scale)))
            if patience is not None:
                patience = max(1, int(round(patience * scale)))
            if verbose:
                print(f"Step-budget scaling: epochs={epochs}, patience={patience} "
                      f"(x{scale:g} for batch {batch_size})", flush=True)

        # 1. To polars + minimal column set + drop NaNs
        df_pl = (
            to_polars(df)
              .select([pl.col(seq_col).alias("seq"), pl.col(rt_col).cast(pl.Float64).alias("rt")])
              .drop_nulls()
              .filter(pl.col("seq").str.len_chars() > 0)
        )
        if val_df is not None:
            val_pl = (
                to_polars(val_df)
                  .select([pl.col(seq_col).alias("seq"), pl.col(rt_col).cast(pl.Float64).alias("rt")])
                  .drop_nulls()
                  .filter(pl.col("seq").str.len_chars() > 0)
            )
            train_pl = df_pl
        else:
            train_pl, val_pl = _split_train_val(df_pl, val_fraction=val_fraction, seed=seed)
        if verbose:
            print(f"Train: {len(train_pl)}, Val: {len(val_pl)}", flush=True)

        # 2. Discover unknown tokens against current vocab and extend the model
        unified = bool(getattr(self.model, "unified_termini", False))
        all_seqs = pl.concat([train_pl, val_pl])["seq"].to_list()
        new_main, new_nterm = discover_tokens(all_seqs, self.vocab, terminal_in_seq=unified)
        fresh_set = set(fresh_mods)
        new_main_triples = [
            (token, parent, token in fresh_set) for token, parent in new_main.items()
        ]
        if new_main_triples or new_nterm:
            if verbose:
                if new_main_triples:
                    print(f"  Discovered {len(new_main_triples)} new main token(s): "
                          f"{[t for t,_,_ in new_main_triples]}")
                if new_nterm:
                    print(f"  Discovered {len(new_nterm)} new N-term mod(s): {sorted(new_nterm)}")
            self.model.extend_vocab(
                new_main_tokens=new_main_triples,
                new_nterm_mods=sorted(new_nterm),
            )

        # 3. Fit RT scale on TRAIN only; build datasets
        train_rts = train_pl["rt"].to_numpy().astype("float32")
        rt_med, rt_mad = fit_rt_scale(train_rts)
        if verbose:
            print(f"  RT scale: median={rt_med:.3f}, MAD={rt_mad:.3f}", flush=True)

        train_ds = RTDataset(train_pl, self.vocab, seq_col="seq", rt_col="rt",
                             rt_median=rt_med, rt_mad=rt_mad, terminal_in_seq=unified)
        val_ds = RTDataset(val_pl, self.vocab, seq_col="seq", rt_col="rt",
                           rt_median=rt_med, rt_mad=rt_mad, terminal_in_seq=unified)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

        # 4. Move model to device, build optimizers (after extend_vocab so they
        # see the resized parameter tensors) and the training criterion.
        self.model.to(device_t)

        if loss == "huber":
            criterion: nn.Module = nn.HuberLoss(delta=1.0)
        elif loss == "mae":
            criterion = nn.L1Loss()
        elif loss == "mixture":
            criterion = LaplaceUniformMixture(init_b=1.0, init_eps=0.05, W=10.0)
        elif loss == "normal_mixture":
            criterion = NormalNormalMixture(init_sigma_clean=0.4, init_sigma_ratio=2.0, init_eps=0.05)
        else:
            raise ValueError(f"Unknown loss {loss!r}; expected huber, mae, mixture, normal_mixture")
        criterion = criterion.to(device_t)

        opts, scheds = _make_optimizers(
            self.model, lr=lr, weight_decay=weight_decay,
            t_max=max(1, epochs - warmup_epochs),
        )
        # Mixture loss carries learnable scalars (b, eps). Fold them into the
        # AdamW group so they're updated by the optimizer.
        loss_params = [p for p in criterion.parameters() if p.requires_grad]
        if loss_params:
            if opts:
                opts[0].add_param_group({"params": loss_params})
            else:
                opt = torch.optim.AdamW(loss_params, lr=lr, weight_decay=weight_decay)
                opts.append(opt)
                scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, epochs - warmup_epochs),
                ))

        # Validation metrics always use MAE so val_loss is cross-loss comparable.
        val_criterion = nn.L1Loss().to(device_t)

        if selection_metric not in ("mae", "r2"):
            raise ValueError(f"Unknown selection_metric {selection_metric!r}; expected mae or r2")

        # Optional TB logging
        writer = None
        if log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)

        # 5. Train
        best_val_r2 = -float("inf")
        best_val_mae = float("inf")
        best_epoch = -1
        best_state: Optional[dict] = None
        wait = 0
        history: list[dict] = []

        for epoch in range(epochs):
            if epoch < warmup_epochs:
                scale = (epoch + 1) / max(1, warmup_epochs)
                for o in opts:
                    base_lr = o.defaults["lr"]
                    for g in o.param_groups:
                        g["lr"] = base_lr * scale

            self.model.train()
            running_loss = 0.0; n = 0
            for seqs, nterm, cterm, targets in train_loader:
                seqs = seqs.to(device_t); nterm = nterm.to(device_t)
                cterm = cterm.to(device_t); targets = targets.to(device_t)
                preds = self.model(seqs, nterm, cterm)
                loss = criterion(preds, targets)
                for o in opts: o.zero_grad()
                loss.backward()
                for o in opts: o.step()
                running_loss += loss.item() * len(targets); n += len(targets)
            train_loss = running_loss / max(1, n)

            if epoch >= warmup_epochs:
                for s in scheds:
                    s.step()

            val_loss, val_mae, val_r2, val_d95 = _eval_metrics(
                self.model, val_loader, val_criterion, device_t,
            )
            current_lr = opts[0].param_groups[0]["lr"] if opts else 0.0
            history.append({
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "val_mae": val_mae,
                "val_r2": val_r2, "val_d95": val_d95, "lr": current_lr,
            })
            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Metrics/val_MAE", val_mae, epoch)
                writer.add_scalar("Metrics/val_R2", val_r2, epoch)
                writer.add_scalar("Metrics/val_d95", val_d95, epoch)
                writer.add_scalar("LR", current_lr, epoch)
            if verbose:
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | train_loss {train_loss:.4f} | "
                    f"val_loss {val_loss:.4f} | val_MAE {val_mae:.4f} | "
                    f"val_R2 {val_r2:.4f} | val_d95 {val_d95:.4f} | LR {current_lr:.2e}",
                    flush=True,
                )

            if selection_metric == "mae":
                improved = val_mae < best_val_mae
            else:  # "r2"
                improved = val_r2 > best_val_r2
            if improved:
                best_val_r2 = val_r2
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if patience is not None and wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1} "
                              f"(no val_{selection_metric.upper()} improvement for {patience} epochs)",
                              flush=True)
                    break

        if writer is not None:
            writer.close()

        # 6. Restore best, persist scale + provenance
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.rt_median = float(rt_med)
        self.rt_mad = float(rt_mad)
        if trained_on is not None:
            self.trained_on = trained_on

        return FitResult(
            best_val_r2=best_val_r2,
            best_val_mae=best_val_mae,
            best_epoch=best_epoch,
            history=history,
        )
