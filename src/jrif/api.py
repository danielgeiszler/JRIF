"""Public DataFrame-driven API for JRIF.

`RTPredictor` is the single user-facing entrypoint. Build one of three ways:

  1. `RTPredictor.from_blank()`  — fresh model with the default vocab.
  2. `RTPredictor.from_checkpoint(path)` — reload a previously-trained model
     (self-describing checkpoint: weights + vocab + RT scale + warp).
  3. After `fit`, the same instance carries a trained model with vocab and
     RT scale already wired up, ready for `.predict(df)` and `.save(path)`.

Calling `.fit(df)` will:
  - discover any modification tokens in the input that the model's vocab does
    not know yet,
  - log every modification in the data and whether it is new,
  - extend the embedding tables (a new residue mod starts from its residue's
    row; a new terminal mod has no parent and is learned from scratch),
  - robust z-score the RT column (median / MAD on the train split),
  - train (Muon for conv weights, AdamW elsewhere; cosine + linear warmup;
    optional early stopping on val R²), and
  - keep the best-val-R² state in `self.model`.

Either polars or pandas DataFrames are accepted; everything internal is polars.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jrif.dataset import (
    PREDICT_CHUNK_SIZE,
    RTDataset,
    Vocabulary,
    discover_tokens,
    drop_untokenizable,
    fit_rt_scale,
    mod_census,
    to_polars,
    tokenize_many,
    unique_rows,
)
from jrif.loss import LaplaceUniformMixture, NormalNormalMixture
from jrif.model import RTModel
from jrif.muon import Muon


# 3: terminus tokens (<nterm>/<cterm>) in every tokenized peptide. Earlier
# formats used a different input representation and cannot be loaded.
CHECKPOINT_FORMAT_VERSION = 3


def _load_checkpoint_dict(path: Path, kind: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    version = ckpt.get("format_version") if isinstance(ckpt, dict) else None
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path}: checkpoint format {version!r} is not supported (expected "
            f"{CHECKPOINT_FORMAT_VERSION}); retrain the model with this version of jrif")
    if ckpt.get("kind", "rt") != kind:
        raise ValueError(f"{path} is not an {kind.upper()} checkpoint")
    return ckpt


# Architecture registry: name -> build_fn(vocab, embedding_dim).
# Only architectures listed here can be saved to / loaded from a checkpoint.
def _build_dilated_cnn(vocab: Vocabulary, embedding_dim: int) -> nn.Module:
    return RTModel(vocab, embedding_dim=embedding_dim)


_ARCH_BUILDERS = {
    "dilated_cnn": _build_dilated_cnn,
}
DEFAULT_ARCH = "dilated_cnn"


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


def _log_mods(seqs, vocab: Vocabulary) -> None:
    """One log block listing every modification spelling in the data, the
    vocabulary token it resolves to, and whether that token is new."""
    census = mod_census(seqs, vocab)
    if not census:
        print("jrif: no modifications in data", flush=True)
        return
    n_new = len({tok for _, _, tok, new, _ in census if new})
    print(f"jrif: {len(census)} modification(s) in data, {n_new} new:", flush=True)
    width = max(len(t) for t, *_ in census)
    for spelling, n, tok, new, parent in census:
        if tok is None:
            status = "AMBIGUOUS (matches several model tokens), skipped"
        else:
            status = "known" if not new else (
                f"NEW, initialized from {parent}" if parent else "NEW, learned from scratch")
            if tok != spelling:
                status += f" (as {tok})"
        print(f"  {spelling:<{width}}  {n:>8,} IDs  {status}", flush=True)


def _calibration_frame(df: DataFrameLike, seq_col: str,
                       numeric: dict[str, str], key: list[str]) -> pl.DataFrame:
    """calibrate() input -> (row, seq, *numeric) with `numeric` mapping
    output name -> input column (cast to Float64). row is the position in
    `df`. Null/empty rows are dropped, then rows are deduplicated on `key`
    keeping the first, so which duplicate survives is deterministic."""
    return (
        to_polars(df)
          .with_row_index("__jrif_row__")
          .select([pl.col("__jrif_row__").alias("row"),
                   pl.col(seq_col).alias("seq"),
                   *[pl.col(src).cast(pl.Float64).alias(dst)
                     for dst, src in numeric.items()]])
          .drop_nulls()
          .filter(pl.col("seq").str.len_chars() > 0)
          .unique(subset=key, keep="first", maintain_order=True)
    )


def _calibration_input(df_pl: pl.DataFrame, vocab: Vocabulary,
                       verbose: bool) -> pl.DataFrame:
    """Drop rows that cannot be tokenized (one log line), then log every mod."""
    df_pl, n_bad, first = drop_untokenizable(df_pl, vocab, will_extend=True)
    if n_bad:
        print(f"jrif: skipped {n_bad} ID(s) that could not be tokenized "
              f"(first: {first})", flush=True)
    if verbose:
        _log_mods(df_pl["seq"].to_list(), vocab)
    return df_pl


def _holdout_split(df_pl: pl.DataFrame, holdout_fraction: float, seed: int,
                   ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Seeded (holdout, calibration) split of a cleaned calibration frame.
    The frame's order is deterministic (_calibration_frame), so the same IDs
    and seed always give the same split; report["holdout"]["row"] recovers it."""
    n = len(df_pl)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    n_hold = max(10, int(n * holdout_fraction))
    return df_pl[order[:n_hold]], df_pl[order[n_hold:]]


def _extend_for(model: nn.Module, seqs) -> None:
    """Add every token in `seqs` the vocabulary lacks. A new residue mod
    starts from its residue's row; a new terminal mod has no parent and is
    learned from scratch."""
    new = discover_tokens(seqs, model.vocab)
    if new:
        model.extend_vocab([(t, p, p is None) for t, p in new.items()])


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
        for seqs, targets in loader:
            seqs = seqs.to(device); targets = targets.to(device)
            preds = model(seqs)
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
        # Optional isotonic output warp (set by calibrate()): piecewise-linear
        # map from raw model output to the calibration data's RT units.
        self.rt_warp: Optional[tuple[np.ndarray, np.ndarray]] = None

    @property
    def vocab(self) -> Vocabulary:
        return self.model.vocab

    # ---- constructors ----

    @classmethod
    def from_blank(
        cls, *, embedding_dim: int = 16, arch: str = DEFAULT_ARCH,
    ) -> "RTPredictor":
        if arch not in _ARCH_BUILDERS:
            raise ValueError(f"Unknown arch {arch!r}. Available: {sorted(_ARCH_BUILDERS)}")
        model = _ARCH_BUILDERS[arch](Vocabulary.default(), embedding_dim)
        return cls(model=model, arch=arch)

    @classmethod
    def from_checkpoint(
        cls, path: Union[str, Path], *, device: Optional[str] = None,
    ) -> "RTPredictor":
        ckpt = _load_checkpoint_dict(Path(path), "rt")
        arch = ckpt["arch"]
        if arch not in _ARCH_BUILDERS:
            raise ValueError(f"Checkpoint arch {arch!r} not supported. "
                             f"Available: {sorted(_ARCH_BUILDERS)}")
        model = _ARCH_BUILDERS[arch](Vocabulary.from_dict(ckpt["vocab"]),
                                     int(ckpt["embedding_dim"]))
        model.load_state_dict(ckpt["model_state_dict"])
        predictor = cls(
            model=model,
            rt_median=float(ckpt["rt_median"]),
            rt_mad=float(ckpt["rt_mad"]),
            trained_on=ckpt.get("trained_on"),
            arch=arch,
        )
        warp = ckpt.get("rt_warp")
        if warp is not None:
            predictor.rt_warp = (np.asarray(warp[0], dtype=float),
                                 np.asarray(warp[1], dtype=float))
        if device is not None:
            predictor.model.to(device)
        return predictor

    # ---- save ----

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "kind": "rt",
            "arch": self.arch,
            "model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "vocab": self.vocab.to_dict(),
            "embedding_dim": self.model.embedding_dim,
            "rt_median": self.rt_median,
            "rt_mad": self.rt_mad,
            "rt_warp": ([self.rt_warp[0].tolist(), self.rt_warp[1].tolist()]
                        if self.rt_warp is not None else None),
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
        """Predict RT for every row of `df`, in input order.

        Each distinct peptide runs through the model once and its prediction
        is broadcast back to its rows; tokenization proceeds PREDICT_CHUNK_SIZE
        peptides at a time, so peak memory does not grow with len(df).
        Raises TokenizationError for any row that cannot be represented.
        """
        device_t = torch.device(
            device if device else _auto_device()
        )
        self.model.to(device_t)
        self.model.eval()

        seqs, inverse = unique_rows(to_polars(df).select(seq_col), [seq_col])
        z = np.empty(seqs.height, dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, seqs.height, PREDICT_CHUNK_SIZE):
                tok = torch.from_numpy(tokenize_many(
                    seqs[seq_col].slice(start, PREDICT_CHUNK_SIZE).to_list(), self.vocab))
                for b in range(0, len(tok), batch_size):
                    zb = self.model(tok[b:b + batch_size].to(device_t)).cpu().numpy()
                    z[start + b:start + b + len(zb)] = zb
        out = z * self.rt_mad + self.rt_median
        if self.rt_warp is not None:
            out = np.interp(out, *self.rt_warp)
        return out[inverse]

    # ---- calibrate ----

    def calibrate(
        self,
        df: DataFrameLike,
        *,
        seq_col: str = "seq",
        rt_col: str = "rt",
        device: Optional[str] = None,
        min_rank: float = 0.95,
        holdout_fraction: float = 0.2,
        seed: int = 42,
        verbose: bool = True,
        curriculum: bool = False,
        **fit_kwargs,
    ) -> dict:
        """Adapt this predictor to one experiment's identified peptides.

        The monotone (isotonic) output warp is ALWAYS fitted first: it maps
        the user's observed RTs into the coordinate system the base model
        already speaks, so any subsequent fine-tuning spends its gradient on
        chemistry, never on re-learning the gradient shape. The warp is
        abandoned only when it demonstrably fails — the frozen model cannot
        rank a held-out fraction of the user's own IDs (e.g. transferring a
        PSMtag experiment onto an LF base: rank 0.79 vs the healthy 0.99;
        results/night/regime2_isotonic.json) — in which case calibrate()
        falls back to plain fine-tuning on raw targets.

        curriculum=True stages adaptation when the data contains modification
        tokens the base has never seen: calibrate on vocabulary-complete
        peptides first, then extend the vocab and continue fine-tuning on the
        full set. No measured benefit over single-phase on simulated novel
        mods (single-phase inheritance init already absorbs them); provided
        as an option for evaluation on real novel-mod experiments.

        When the warp holds, fine-tuning runs on inverse-warped targets and
        the warp is re-fitted afterwards; warp-only and warp+fine-tune are
        then compared on the holdout and the better one is kept, so tiny-ID
        experiments (where training overfits) automatically resolve to
        warp-only. Returns a report dict: mode is one of "warp",
        "warp+finetune", "finetune_raw".

        report["holdout"] is a polars frame (row, seq, rt, pred): row is the
        position in `df` of each held-out ID, pred is what holdout_mae was
        computed from. The split is deterministic: input rows are
        deduplicated on seq keeping the first occurrence, then shuffled with
        `seed`. The holdout drives the decisions above (rank gate, epoch
        selection, warp vs warp+finetune), so holdout_mae is a selection
        score; score IDs never passed to calibrate() for an unbiased one.
        """
        df_pl = _calibration_frame(df, seq_col, {"rt": rt_col}, key=["seq"])
        df_pl = _calibration_input(df_pl, self.vocab, verbose)
        run = self._calibrate_curriculum if curriculum else self._calibrate_single
        return run(df_pl, device=device, min_rank=min_rank,
                   holdout_fraction=holdout_fraction, seed=seed,
                   verbose=verbose, **fit_kwargs)

    def _calibrate_single(self, df_pl: pl.DataFrame, *, device, min_rank,
                          holdout_fraction, seed, verbose, **fit_kwargs) -> dict:
        """Single-phase calibrate on a cleaned _calibration_frame (row, seq, rt)."""
        from scipy.stats import spearmanr
        from sklearn.isotonic import IsotonicRegression

        # New tokens are added before anything is scored, so the frozen-base
        # rank check and warp see every ID (a new residue mod starts as its
        # residue) and the base-weight snapshot below matches the model shape.
        _extend_for(self.model, df_pl["seq"].to_list())
        n = len(df_pl)
        hold, cal = _holdout_split(df_pl, holdout_fraction, seed)
        y_hold = hold["rt"].to_numpy()

        self.rt_warp = None
        z_hold = self.predict(hold, seq_col="seq", device=device)
        rank = float(spearmanr(y_hold, z_hold).statistic)
        report = {"n": n, "holdout_rank": rank}

        if rank < min_rank:
            # Warp genuinely failed: the base cannot order these peptides, so
            # no monotone map of its output can help. Plain fine-tune.
            if verbose:
                print(f"calibrate: warp unusable (holdout rank {rank:.4f} < "
                      f"{min_rank}); fine-tuning on raw targets", flush=True)
            fit_kwargs.setdefault("lr", FINETUNE_LR)
            fit_kwargs.setdefault("verbose", verbose)
            # train on cal only: the holdout is scored below
            self.fit(cal, seq_col="seq", rt_col="rt", val_df=hold,
                     device=device, seed=seed, **{**SHIPPING_RECIPE, **fit_kwargs})
            pred = self.predict(hold, seq_col="seq", device=device)
            report["holdout_mae"] = float(np.abs(pred - y_hold).mean())
            report["holdout"] = hold.with_columns(pl.Series("pred", pred))
            report["mode"] = "finetune_raw"
            return report

        def fit_warp(z, y):
            iso = IsotonicRegression(out_of_bounds="clip").fit(z, y)
            xk = np.asarray(iso.X_thresholds_, dtype=float)
            yk = np.asarray(iso.y_thresholds_, dtype=float)
            return xk, yk

        # 1. warp from calibration split; score warp-only on holdout
        z_cal = self.predict(cal, seq_col="seq", device=device)
        xk, yk = fit_warp(z_cal, cal["rt"].to_numpy())
        warp_hold_mae = float(np.abs(np.interp(z_hold, xk, yk) - y_hold).mean())

        # 2. fine-tune on inverse-warped targets (strictly-increasing knots
        #    so the pseudo-inverse is well defined across flat segments)
        keep = np.concatenate([[True], np.diff(yk) > 0])
        inv_x, inv_y = yk[keep], xk[keep]
        inv_targets = np.interp(cal["rt"].to_numpy(), inv_x, inv_y)
        hold_inv = hold.with_columns(pl.Series("rt", np.interp(y_hold, inv_x, inv_y)))
        base_state = copy.deepcopy(self.model.state_dict())
        fit_kwargs.setdefault("lr", FINETUNE_LR)
        fit_kwargs.setdefault("verbose", False)
        self.fit(cal.with_columns(pl.Series("rt", inv_targets)),
                 seq_col="seq", rt_col="rt", val_df=hold_inv,
                 device=device, seed=seed, **{**SHIPPING_RECIPE, **fit_kwargs})
        # refit warp on the fine-tuned model's own outputs
        z_cal2 = self.predict(cal, seq_col="seq", device=device)
        xk2, yk2 = fit_warp(z_cal2, cal["rt"].to_numpy())
        z_hold2 = self.predict(hold, seq_col="seq", device=device)
        ft_hold_mae = float(np.abs(np.interp(z_hold2, xk2, yk2) - y_hold).mean())

        report["warp_only_holdout_mae"] = warp_hold_mae
        report["warp_finetune_holdout_mae"] = ft_hold_mae
        if ft_hold_mae <= warp_hold_mae:
            # keep fine-tuned weights; final warp refit on ALL ids
            z_all = self.predict(df_pl, seq_col="seq", device=device)
            self.rt_warp = fit_warp(z_all, df_pl["rt"].to_numpy())
            report["mode"] = "warp+finetune"
            report["holdout_mae"] = ft_hold_mae
            hold_pred = np.interp(z_hold2, xk2, yk2)
        else:
            # training hurt (tiny N): restore base weights, warp-only on ALL ids
            self.model.load_state_dict(base_state)
            z_all = self.predict(df_pl, seq_col="seq", device=device)
            self.rt_warp = fit_warp(z_all, df_pl["rt"].to_numpy())
            report["mode"] = "warp"
            report["holdout_mae"] = warp_hold_mae
            hold_pred = np.interp(z_hold, xk, yk)
        report["holdout"] = hold.with_columns(pl.Series("pred", hold_pred))
        if verbose:
            print(f"calibrate: {report['mode']} (holdout rank {rank:.4f}; "
                  f"warp-only {warp_hold_mae:.4f} vs warp+ft {ft_hold_mae:.4f})", flush=True)
        return report

    def _calibrate_curriculum(self, df_pl: pl.DataFrame, *, device,
                              min_rank, holdout_fraction, seed, verbose,
                              **fit_kwargs) -> dict:
        """Stage 1: calibrate on vocabulary-complete peptides. Stage 2:
        extend vocab and continue fine-tuning on the full set through the
        stage-1 warp; refit the warp. Falls back to single-phase when there
        are no novel tokens or too few clean peptides. The report's holdout
        and holdout_mae are stage 1's; stage 2 trains on those rows."""
        from sklearn.isotonic import IsotonicRegression

        novel = set(discover_tokens(df_pl["seq"].to_list(), self.vocab))
        if novel:
            def has_novel(seq: str) -> bool:
                return bool(discover_tokens([seq], self.vocab))
            is_novel = df_pl["seq"].map_elements(has_novel, return_dtype=pl.Boolean)
            clean = df_pl.filter(~is_novel)
        else:
            clean = df_pl
        if not novel or len(clean) < 50:
            if verbose and novel:
                print("calibrate(curriculum): too few vocabulary-complete "
                      "peptides; using single-phase", flush=True)
            return self._calibrate_single(df_pl, device=device, min_rank=min_rank,
                                          holdout_fraction=holdout_fraction, seed=seed,
                                          verbose=verbose, **fit_kwargs)

        report = self._calibrate_single(clean, device=device, min_rank=min_rank,
                                        holdout_fraction=holdout_fraction, seed=seed,
                                        verbose=verbose, **fit_kwargs)
        stage1_mode = report["mode"]
        if self.rt_warp is None:
            # stage 1 fell back to raw fine-tuning; continue single-phase on all
            fit_kwargs.setdefault("lr", FINETUNE_LR)
            fit_kwargs.setdefault("verbose", verbose)
            self.fit(df_pl, seq_col="seq", rt_col="rt", device=device,
                     seed=seed, **{**SHIPPING_RECIPE, **fit_kwargs})
            report.update(mode="curriculum_raw", stage1_mode=stage1_mode)
            return report

        xk, yk = self.rt_warp
        keep = np.concatenate([[True], np.diff(yk) > 0])
        inv_targets = np.interp(df_pl["rt"].to_numpy(), yk[keep], xk[keep])
        fit_kwargs.setdefault("lr", FINETUNE_LR)
        fit_kwargs.setdefault("verbose", False)
        self.fit(df_pl.with_columns(pl.Series("rt", inv_targets)),
                 seq_col="seq", rt_col="rt", device=device, seed=seed,
                 **{**SHIPPING_RECIPE, **fit_kwargs})
        z_all = self.predict(df_pl, seq_col="seq", device=device)
        iso = IsotonicRegression(out_of_bounds="clip").fit(z_all, df_pl["rt"].to_numpy())
        self.rt_warp = (np.asarray(iso.X_thresholds_, float),
                        np.asarray(iso.y_thresholds_, float))
        report.update(mode="curriculum", stage1_mode=stage1_mode,
                      n_novel_tokens=len(novel), n=len(df_pl))
        if verbose:
            print(f"calibrate(curriculum): stage1={stage1_mode} on "
                  f"{len(clean)} clean IDs; stage2 on {len(df_pl)} with "
                  f"{len(novel)} new token(s)", flush=True)
        return report

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
        seed: int = 42,
        device: Optional[str] = None,
        num_threads: Optional[int] = None,
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
        self.rt_warp = None  # stale after any weight update
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

        train_pl, n_bad_tr, first = drop_untokenizable(train_pl, self.vocab, will_extend=True)
        val_pl, n_bad_va, first_va = drop_untokenizable(val_pl, self.vocab, will_extend=True)
        if n_bad_tr + n_bad_va:
            print(f"jrif: skipped {n_bad_tr + n_bad_va} ID(s) that could not "
                  f"be tokenized (first: {first or first_va})", flush=True)
        if verbose:
            print(f"Train: {len(train_pl)}, Val: {len(val_pl)}", flush=True)

        # 2. Discover unknown tokens against current vocab and extend the model
        all_seqs = pl.concat([train_pl, val_pl])["seq"].to_list()
        if verbose:
            _log_mods(all_seqs, self.vocab)
        _extend_for(self.model, all_seqs)

        # 3. Fit RT scale on TRAIN only; build datasets
        train_rts = train_pl["rt"].to_numpy().astype("float32")
        rt_med, rt_mad = fit_rt_scale(train_rts)
        if verbose:
            print(f"  RT scale: median={rt_med:.3f}, MAD={rt_mad:.3f}", flush=True)

        train_ds = RTDataset(train_pl, self.vocab, seq_col="seq", rt_col="rt",
                             rt_median=rt_med, rt_mad=rt_mad)
        val_ds = RTDataset(val_pl, self.vocab, seq_col="seq", rt_col="rt",
                           rt_median=rt_med, rt_mad=rt_mad)
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
            for seqs, targets in train_loader:
                seqs = seqs.to(device_t); targets = targets.to(device_t)
                preds = self.model(seqs)
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
