"""Apples-to-apples comparison: dilated-CNN (RTModel) vs timePlex on PSMtag.

Pipeline per architecture:
  1. Pretrain from scratch on the DIA-NN label-free library
     (data/DIANNv1p9_JD0507_509_LF_HY.tsv).
  2. Fine-tune on the PSMtag library (data/psmtag_library.tsv).
  3. Evaluate on the held-out PSMtag test split.

Both architectures use the same:
  - peptide-level 80/10/10 split (seed=42)
  - tokenization (JRIF Vocabulary, MAX_LEN=50)
  - robust-z RT normalization
  - optimizer (AdamW only — no Muon, which is dilated-CNN-specific)
  - loss (HuberLoss, delta=1.0)
  - schedule (cosine + 3-epoch linear warmup)
  - epochs / patience
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jrif.dataset import (
    RTDataset, Vocabulary,
    discover_tokens, fit_rt_scale,
    load_psmtag_library, load_spectral_library,
)
from jrif.model import RTModel
from jrif.model_timeplex import TimePlexModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(train_df, val_df, test_df, vocab, batch_size, seed):
    rt_med, rt_mad = fit_rt_scale(train_df["rt"].to_numpy().astype("float32"))
    train_ds = RTDataset(train_df, vocab, rt_median=rt_med, rt_mad=rt_mad)
    val_ds   = RTDataset(val_df,   vocab, rt_median=rt_med, rt_mad=rt_mad)
    test_ds  = RTDataset(test_df,  vocab, rt_median=rt_med, rt_mad=rt_mad)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader, test_loader, rt_med, rt_mad


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


def train_loop(
    model, train_loader, val_loader, *,
    epochs, lr, weight_decay, warmup_epochs, patience, device, label, log_every=10,
):
    """Train with AdamW + cosine + warmup + Huber. Returns (best_state, best_metrics, history)."""
    criterion = nn.HuberLoss(delta=1.0)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs))

    best_r2 = -float("inf"); best_state = None; best_epoch = -1
    best_mae = float("inf"); wait = 0
    history = []
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
        train_loss = running / n
        if epoch >= warmup_epochs:
            sched.step()

        val_loss, val_mae, val_r2, val_d95 = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_mae": val_mae, "val_r2": val_r2, "val_d95": val_d95,
                        "lr": opt.param_groups[0]["lr"]})

        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  [{label}] ep {epoch+1:3d}/{epochs} | "
                  f"train {train_loss:.4f} | val_loss {val_loss:.4f} | val_R2 {val_r2:.4f} | "
                  f"val_MAE {val_mae:.4f} | LR {opt.param_groups[0]['lr']:.2e}")

        if val_r2 > best_r2:
            best_r2 = val_r2; best_mae = val_mae; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{label}] early stop at epoch {epoch+1} (no val_R2 improvement for {patience})")
                break

    return best_state, {"best_r2": best_r2, "best_mae": best_mae, "best_epoch": best_epoch}, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lf-data",     default="data/DIANNv1p9_JD0507_509_LF_HY.tsv")
    p.add_argument("--psmtag-data", default="data/psmtag_library.tsv")
    p.add_argument("--pretrain-epochs", type=int, default=100)
    p.add_argument("--pretrain-lr",     type=float, default=1e-3)
    p.add_argument("--finetune-epochs", type=int, default=50)
    p.add_argument("--finetune-lr",     type=float, default=1e-4)
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument("--warmup-epochs",   type=int, default=3)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--patience",        type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Pretrain: {args.pretrain_epochs} ep, lr={args.pretrain_lr}")
    print(f"Finetune: {args.finetune_epochs} ep, lr={args.finetune_lr}")
    print(f"Batch={args.batch_size}, warmup={args.warmup_epochs}, wd={args.weight_decay}, patience={args.patience}, seed={args.seed}")

    # Phase 0: load data once
    print("\n[data] Loading label-free DIA-NN library...")
    lf_train, lf_val, lf_test = load_spectral_library(args.lf_data, seed=args.seed)
    print(f"  LF train/val/test: {len(lf_train)}/{len(lf_val)}/{len(lf_test)}")
    print("[data] Loading PSMtag library...")
    pt_train, pt_val, pt_test = load_psmtag_library(args.psmtag_data, seed=args.seed)
    print(f"  PSMtag train/val/test: {len(pt_train)}/{len(pt_val)}/{len(pt_test)}")

    # Each arch gets its own pre-extended vocab built on combined LF+PSMtag tokens,
    # so both share the same input space and same one-hot dimension. (Identity for
    # dilated CNN; relevant for timePlex which one-hots VOCAB_SIZE.)
    combined_seqs = (lf_train["seq"].to_list() + pt_train["seq"].to_list()
                     + lf_val["seq"].to_list() + pt_val["seq"].to_list())

    results = {}
    for arch_name, build_model in [
        ("dilated_cnn", lambda v: RTModel(v, embedding_dim=16)),
        ("timeplex",    lambda v: TimePlexModel(v)),
    ]:
        print(f"\n{'='*72}\nARCH: {arch_name}\n{'='*72}")
        set_seed(args.seed)

        vocab = Vocabulary.default()
        new_main, new_nterm = discover_tokens(combined_seqs, vocab)
        if new_main:
            print(f"  Pre-extending vocab with {len(new_main)} new main tokens")
        for tok, parent in new_main.items():
            vocab.extend(tok, parent, fresh=False)
        for mod in sorted(new_nterm):
            vocab.extend_nterm_mod(mod)

        model = build_model(vocab).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {n_params:,}")

        # Phase 1: pretrain on label-free DIA-NN library
        print(f"\n--- [{arch_name}] Phase 1: pretrain on label-free DIA-NN library ---")
        t0 = time.time()
        train_loader, val_loader, _test_loader_lf, _, _ = make_loaders(
            lf_train, lf_val, lf_test, vocab, args.batch_size, args.seed,
        )
        pre_state, pre_metrics, _pre_hist = train_loop(
            model, train_loader, val_loader,
            epochs=args.pretrain_epochs, lr=args.pretrain_lr,
            weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
            patience=args.patience, device=device, label=f"{arch_name}/pretrain",
        )
        pre_time = time.time() - t0
        print(f"  pretrain best val R2={pre_metrics['best_r2']:.4f} "
              f"MAE={pre_metrics['best_mae']:.4f} at epoch {pre_metrics['best_epoch']+1}  "
              f"({pre_time:.1f}s)")
        model.load_state_dict(pre_state)

        # Phase 2: fine-tune on PSMtag library
        print(f"\n--- [{arch_name}] Phase 2: fine-tune on PSMtag library ---")
        t1 = time.time()
        train_loader, val_loader, test_loader, rt_med, rt_mad = make_loaders(
            pt_train, pt_val, pt_test, vocab, args.batch_size, args.seed,
        )
        ft_state, ft_metrics, _ft_hist = train_loop(
            model, train_loader, val_loader,
            epochs=args.finetune_epochs, lr=args.finetune_lr,
            weight_decay=args.weight_decay, warmup_epochs=0,  # no warmup for FT
            patience=args.patience, device=device, label=f"{arch_name}/finetune",
        )
        ft_time = time.time() - t1
        print(f"  finetune best val R2={ft_metrics['best_r2']:.4f} "
              f"MAE={ft_metrics['best_mae']:.4f} at epoch {ft_metrics['best_epoch']+1}  "
              f"({ft_time:.1f}s)")
        model.load_state_dict(ft_state)

        # Final test eval (z-space)
        criterion = nn.HuberLoss(delta=1.0)
        test_loss, test_mae, test_r2, test_d95 = evaluate(model, test_loader, criterion, device)
        print(f"  [TEST/{arch_name}] R2={test_r2:.4f}  MAE={test_mae:.4f}  d95={test_d95:.4f}")

        results[arch_name] = {
            "params": n_params,
            "pretrain_best_r2": pre_metrics["best_r2"],
            "pretrain_best_mae": pre_metrics["best_mae"],
            "pretrain_best_epoch": pre_metrics["best_epoch"] + 1,
            "pretrain_time_s": pre_time,
            "finetune_best_r2": ft_metrics["best_r2"],
            "finetune_best_mae": ft_metrics["best_mae"],
            "finetune_best_epoch": ft_metrics["best_epoch"] + 1,
            "finetune_time_s": ft_time,
            "test_r2": test_r2, "test_mae": test_mae, "test_d95": test_d95,
        }

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"{'arch':<14} {'params':>10} {'pre R2':>9} {'pre MAE':>9} {'ft R2':>9} {'ft MAE':>9} {'TEST R2':>9} {'TEST MAE':>10} {'TEST d95':>10}")
    for arch_name, r in results.items():
        print(
            f"{arch_name:<14} {r['params']:>10,} "
            f"{r['pretrain_best_r2']:>9.4f} {r['pretrain_best_mae']:>9.4f} "
            f"{r['finetune_best_r2']:>9.4f} {r['finetune_best_mae']:>9.4f} "
            f"{r['test_r2']:>9.4f} {r['test_mae']:>10.4f} {r['test_d95']:>10.4f}"
        )

    out = Path("results/bench_arch_psmtag.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
