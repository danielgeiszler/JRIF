"""Noise-robustness bench: dilated CNN vs timePlex.

Pipeline:
  - Pretrain each architecture once on clean LF DIA-NN library.
  - For each noise level sigma in {0, 0.1, 0.25, 0.5, 1.0}, restore the
    pretrain weights and fine-tune on the PSMtag train split with Gaussian
    additive noise of std = sigma * train_MAD added to every RT label.
  - Evaluate on the CLEAN PSMtag test split (test labels never noised).

The architecture whose test metrics degrade less under increasing fine-tune
noise is the more noise-robust one.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jrif.dataset import (
    RTDataset, Vocabulary, discover_tokens, fit_rt_scale,
    load_psmtag_library, load_spectral_library,
)
from jrif.model import RTModel
from jrif.model_timeplex import TimePlexModel


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_loader(df, vocab, batch_size, shuffle, rt_med, rt_mad, seed=None):
    ds = RTDataset(df, vocab, rt_median=rt_med, rt_mad=rt_mad)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0; preds, tgts = [], []
    with torch.no_grad():
        for seqs, nterm, cterm, targets in loader:
            seqs = seqs.to(device); nterm = nterm.to(device)
            cterm = cterm.to(device); targets = targets.to(device)
            y = model(seqs, nterm, cterm)
            total += criterion(y, targets).item() * len(targets)
            preds.append(y.cpu()); tgts.append(targets.cpu())
    p = torch.cat(preds); t = torch.cat(tgts)
    err = (p - t).abs()
    mae = err.mean().item(); d95 = err.quantile(0.95).item()
    ss_res = ((t - p) ** 2).sum().item(); ss_tot = ((t - t.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return total / len(t), mae, r2, d95


def train_loop(model, train_loader, val_loader, *, epochs, lr, weight_decay,
               warmup_epochs, patience, device, label, log_every=10):
    criterion = nn.HuberLoss(delta=1.0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs))

    best_r2 = -float("inf"); best_mae = float("inf")
    best_state = None; best_epoch = -1; wait = 0
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            for g in opt.param_groups:
                g["lr"] = lr * (epoch + 1) / max(1, warmup_epochs)
        model.train()
        running = 0.0; n = 0
        for seqs, nterm, cterm, targets in train_loader:
            seqs = seqs.to(device); nterm = nterm.to(device)
            cterm = cterm.to(device); targets = targets.to(device)
            y = model(seqs, nterm, cterm)
            loss = criterion(y, targets)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * len(targets); n += len(targets)
        if epoch >= warmup_epochs: sched.step()

        _, val_mae, val_r2, _ = evaluate(model, val_loader, criterion, device)
        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  [{label}] ep {epoch+1:3d}/{epochs} | "
                  f"train {running/max(1,n):.4f} | val_R2 {val_r2:.4f} | val_MAE {val_mae:.4f}")
        if val_r2 > best_r2:
            best_r2 = val_r2; best_mae = val_mae; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{label}] early stop at epoch {epoch+1}")
                break
    return best_state, {"best_r2": best_r2, "best_mae": best_mae, "best_epoch": best_epoch + 1}


def add_gaussian_noise(df: pl.DataFrame, sigma_in_mad: float, train_mad: float, seed: int) -> pl.DataFrame:
    if sigma_in_mad <= 0:
        return df
    rng = np.random.default_rng(seed)
    rts = df["rt"].to_numpy().astype("float32")
    noise = rng.normal(0.0, sigma_in_mad * train_mad, size=rts.shape).astype("float32")
    return df.with_columns(pl.Series("rt", rts + noise))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lf-data",     default="data/DIANNv1p9_JD0507_509_LF_HY.tsv")
    p.add_argument("--psmtag-data", default="data/psmtag_library.tsv")
    p.add_argument("--pretrain-epochs", type=int, default=80)
    p.add_argument("--pretrain-lr",     type=float, default=1e-3)
    p.add_argument("--finetune-epochs", type=int, default=40)
    p.add_argument("--finetune-lr",     type=float, default=1e-4)
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument("--warmup-epochs",   type=int, default=3)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--patience",        type=int, default=15)
    p.add_argument("--noise-sigmas", default="0,0.1,0.25,0.5,1.0",
                   help="Gaussian noise std as multiples of train MAD")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    sigmas = [float(s) for s in args.noise_sigmas.split(",")]
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Noise sigmas (× train MAD): {sigmas}")

    print("\n[data] LF DIA-NN library...")
    lf_train, lf_val, _ = load_spectral_library(args.lf_data, seed=args.seed)
    print(f"  LF train/val: {len(lf_train)}/{len(lf_val)}")
    print("[data] PSMtag library...")
    pt_train, pt_val, pt_test = load_psmtag_library(args.psmtag_data, seed=args.seed)
    print(f"  PSMtag train/val/test: {len(pt_train)}/{len(pt_val)}/{len(pt_test)}")

    combined = (lf_train["seq"].to_list() + pt_train["seq"].to_list()
                + lf_val["seq"].to_list() + pt_val["seq"].to_list())

    results = {}
    for arch_name, build in [
        ("dilated_cnn", lambda v: RTModel(v, embedding_dim=16)),
        ("timeplex",    lambda v: TimePlexModel(v)),
    ]:
        print(f"\n{'='*72}\nARCH: {arch_name}\n{'='*72}")
        set_seed(args.seed)

        vocab = Vocabulary.default()
        new_main, new_nterm = discover_tokens(combined, vocab)
        for tok, parent in new_main.items(): vocab.extend(tok, parent, fresh=False)
        for mod in sorted(new_nterm): vocab.extend_nterm_mod(mod)

        model = build(vocab).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {n_params:,}")

        # Pretrain
        print(f"\n--- [{arch_name}] Pretrain (clean LF) ---")
        t0 = time.time()
        lf_med, lf_mad = fit_rt_scale(lf_train["rt"].to_numpy().astype("float32"))
        train_loader = make_loader(lf_train, vocab, args.batch_size, True, lf_med, lf_mad, args.seed)
        val_loader   = make_loader(lf_val,   vocab, args.batch_size * 2, False, lf_med, lf_mad)
        pre_state, pre_metrics = train_loop(
            model, train_loader, val_loader,
            epochs=args.pretrain_epochs, lr=args.pretrain_lr,
            weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
            patience=args.patience, device=device, label=f"{arch_name}/pre",
        )
        print(f"  pretrain: best val R2={pre_metrics['best_r2']:.4f} MAE={pre_metrics['best_mae']:.4f} "
              f"({time.time()-t0:.1f}s)")

        # Build clean PSMtag loaders/scale once (test always clean)
        ft_med, ft_mad = fit_rt_scale(pt_train["rt"].to_numpy().astype("float32"))
        val_loader  = make_loader(pt_val,  vocab, args.batch_size * 2, False, ft_med, ft_mad)
        test_loader = make_loader(pt_test, vocab, args.batch_size * 2, False, ft_med, ft_mad)

        per_sigma = {}
        for sigma in sigmas:
            print(f"\n--- [{arch_name}] FT with sigma={sigma}*MAD ---")
            t1 = time.time()
            # Restore pretrain snapshot before each FT (deepcopy of state_dict)
            model.load_state_dict({k: v.clone() for k, v in pre_state.items()})

            train_noisy = add_gaussian_noise(pt_train, sigma, ft_mad, seed=args.seed)
            train_loader = make_loader(train_noisy, vocab, args.batch_size, True, ft_med, ft_mad, args.seed)
            ft_state, ft_metrics = train_loop(
                model, train_loader, val_loader,
                epochs=args.finetune_epochs, lr=args.finetune_lr,
                weight_decay=args.weight_decay, warmup_epochs=0,
                patience=args.patience, device=device, label=f"{arch_name}/ft@{sigma}",
            )
            model.load_state_dict(ft_state)
            criterion = nn.HuberLoss(delta=1.0)
            _, test_mae, test_r2, test_d95 = evaluate(model, test_loader, criterion, device)
            print(f"  sigma={sigma}: TEST R2={test_r2:.4f} MAE={test_mae:.4f} d95={test_d95:.4f}  "
                  f"({time.time()-t1:.1f}s)")
            per_sigma[str(sigma)] = {
                "val_r2": ft_metrics["best_r2"], "val_mae": ft_metrics["best_mae"],
                "test_r2": test_r2, "test_mae": test_mae, "test_d95": test_d95,
                "best_ft_epoch": ft_metrics["best_epoch"],
            }
        results[arch_name] = {"params": n_params,
                              "pretrain_r2": pre_metrics["best_r2"],
                              "pretrain_mae": pre_metrics["best_mae"],
                              "noise_sweep": per_sigma}

    print(f"\n{'='*72}\nNOISE-ROBUSTNESS SUMMARY (test set always clean)\n{'='*72}")
    print(f"{'arch':<14} {'sigma':>6}  {'TEST R2':>9} {'TEST MAE':>10} {'TEST d95':>10}")
    for arch_name, r in results.items():
        for sigma in sigmas:
            row = r["noise_sweep"][str(sigma)]
            print(f"{arch_name:<14} {sigma:>6.2f}  {row['test_r2']:>9.4f} "
                  f"{row['test_mae']:>10.4f} {row['test_d95']:>10.4f}")

    print(f"\n{'arch':<14} {'pre R2':>9} | per-sigma TEST MAE: " + "  ".join(f"{s:>6.2f}" for s in sigmas))
    for arch_name, r in results.items():
        cells = "  ".join(f"{r['noise_sweep'][str(s)]['test_mae']:>6.4f}" for s in sigmas)
        print(f"{arch_name:<14} {r['pretrain_r2']:>9.4f} | {cells}")

    out = Path("results/bench_noise_robustness.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
