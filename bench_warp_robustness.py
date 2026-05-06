"""Distribution-warp robustness: dilated CNN vs timePlex vs wide-head dilated.

Tests how well each architecture adapts to a chromatographic shift modeled as
a continuous piecewise-linear warp on RT, applied to the SAME dataset
(PSMtag library) used for pretraining. No LF pretrain — the warp is the only
distribution change between pretrain and fine-tune.

Warp on centred RT x = rt - median:
  x < q1           -> x - q1/2            (left tail pulled toward 0)
  q1 <= x <= q3    -> x / 2               (middle 50% compressed)
  x > q3           -> x - q3/2            (right tail pulled toward 0)
Continuous (slopes 1, 0.5, 1) and computed from the PSMtag train RTs.

Pipeline per architecture:
  1. Pretrain from scratch on UNWARPED PSMtag train. Eval on unwarped test.
  2. Restore pretrain snapshot. Fine-tune on WARPED PSMtag train. Eval on
     warped test. Both val splits used for early stop / best-checkpoint
     selection match the warp state of the corresponding train.
"""

from __future__ import annotations

import argparse
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
    RTDataset, Vocabulary, discover_tokens, fit_rt_scale, load_psmtag_library,
)
from jrif.model import RTModel
from jrif.model_timeplex import TimePlexModel
from jrif.model_xformer import XformerSkipModel
from jrif.model_xformer_iha import XformerSkipIHAModel
from jrif.model_xformer_pure import XformerPureModel


def build_wide_head_dilated_cnn(vocab, embedding_dim: int = 16):
    """Dilated-CNN backbone unchanged; FC head widened from 149->128->1
    to 149->512->256->128->1 (GELU + dropout)."""
    m = RTModel(vocab, embedding_dim=embedding_dim)
    head_in = m.head[1].in_features
    m.head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(head_in, 512), nn.GELU(),
        nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU(),
        nn.Dropout(0.2), nn.Linear(256, 128), nn.GELU(),
        nn.Linear(128, 1),
    )
    return m


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def warp_rt(rt: np.ndarray, q1: float, q3: float, median: float) -> np.ndarray:
    centered = rt - median
    out = centered.astype("float32").copy()
    left   = centered < q1
    middle = (centered >= q1) & (centered <= q3)
    right  = centered > q3
    out[left]   = centered[left]   - q1 / 2.0
    out[middle] = centered[middle] * 0.5
    out[right]  = centered[right]  - q3 / 2.0
    return out


def warp_df(df: pl.DataFrame, q1: float, q3: float, median: float) -> pl.DataFrame:
    rts = df["rt"].to_numpy().astype("float32")
    return df.with_columns(pl.Series("rt", warp_rt(rts, q1, q3, median)))


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
            print(f"  [{label}] ep {epoch+1:3d}/{epochs} | train {running/max(1,n):.4f} | "
                  f"val_R2 {val_r2:.4f} | val_MAE {val_mae:.4f}", flush=True)
        if val_r2 > best_r2:
            best_r2 = val_r2; best_mae = val_mae; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{label}] early stop at epoch {epoch+1}", flush=True)
                break
    return best_state, {"best_r2": best_r2, "best_mae": best_mae,
                        "best_epoch": best_epoch + 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--psmtag-data", default="data/psmtag_library.tsv")
    p.add_argument("--pretrain-epochs", type=int, default=100)
    p.add_argument("--pretrain-lr",     type=float, default=1e-3)
    p.add_argument("--finetune-epochs", type=int, default=40)
    p.add_argument("--finetune-lr",     type=float, default=1e-4)
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument("--warmup-epochs",   type=int, default=3)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--pretrain-patience", type=int, default=25)
    p.add_argument("--finetune-patience", type=int, default=15)
    p.add_argument("--ft-subsample", type=int, default=None,
                   help="If set, randomly sample this many peptides from the warped "
                        "train split for fine-tuning (simulates a small new-dataset "
                        "label set). Test/val splits are unchanged.")
    p.add_argument("--archs", default="xformer_skip",
                   help="Comma-separated list of arch names to run. Available: "
                        "dilated_cnn, dilated_cnn_wide, xformer_skip, "
                        "xformer_skip_iha, timeplex. Defaults to just the current "
                        "incumbent for fast iteration; pass e.g. "
                        "--archs xformer_skip,my_new_arch when proposing alternatives.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    print("\n[data] PSMtag library...", flush=True)
    pt_train, pt_val, pt_test = load_psmtag_library(args.psmtag_data, seed=args.seed)
    print(f"  PSMtag train/val/test: {len(pt_train)}/{len(pt_val)}/{len(pt_test)}", flush=True)

    # Compute warp parameters from PSMtag TRAIN only and reuse on val/test
    pt_train_rts = pt_train["rt"].to_numpy().astype("float32")
    pt_med  = float(np.median(pt_train_rts))
    pt_q1   = float(np.quantile(pt_train_rts - pt_med, 0.25))
    pt_q3   = float(np.quantile(pt_train_rts - pt_med, 0.75))
    print(f"  Warp params (centred): q1={pt_q1:.3f}, q3={pt_q3:.3f}, median={pt_med:.3f}", flush=True)

    pt_train_warped = warp_df(pt_train, pt_q1, pt_q3, pt_med)
    pt_val_warped   = warp_df(pt_val,   pt_q1, pt_q3, pt_med)
    pt_test_warped  = warp_df(pt_test,  pt_q1, pt_q3, pt_med)

    # Optional FT subsample: simulate a small new-dataset label set.
    if args.ft_subsample is not None and args.ft_subsample < len(pt_train_warped):
        pt_train_warped = pt_train_warped.sample(
            n=args.ft_subsample, seed=args.seed, with_replacement=False, shuffle=True,
        )
        print(f"  FT train subsampled to {len(pt_train_warped)} peptides", flush=True)

    orig = pt_train_rts
    wp   = pt_train_warped["rt"].to_numpy()
    print(f"  Original RT range / std: [{orig.min():.2f},{orig.max():.2f}]  std={orig.std():.3f}", flush=True)
    print(f"  Warped   RT range / std: [{wp.min():.2f},{wp.max():.2f}]  std={wp.std():.3f}", flush=True)

    # Vocab built from all sequences (train+val of PSMtag), so each arch sees
    # the same one-hot dimension for timePlex.
    combined = pt_train["seq"].to_list() + pt_val["seq"].to_list()

    arch_registry = {
        "dilated_cnn":         lambda v: RTModel(v, embedding_dim=16),
        "dilated_cnn_wide":    lambda v: build_wide_head_dilated_cnn(v, embedding_dim=16),
        "xformer_skip":        lambda v: XformerSkipModel(v, embedding_dim=16,
                                                            d_model=128, nhead=4,
                                                            num_layers=2, dim_feedforward=512,
                                                            dropout=0.1),
        "xformer_skip_h8":     lambda v: XformerSkipModel(v, embedding_dim=16,
                                                            d_model=128, nhead=8,
                                                            num_layers=2, dim_feedforward=512,
                                                            dropout=0.1),
        "xformer_skip_h2":     lambda v: XformerSkipModel(v, embedding_dim=16,
                                                            d_model=128, nhead=2,
                                                            num_layers=2, dim_feedforward=512,
                                                            dropout=0.1),
        "xformer_skip_3l":     lambda v: XformerSkipModel(v, embedding_dim=16,
                                                            d_model=128, nhead=4,
                                                            num_layers=3, dim_feedforward=512,
                                                            dropout=0.1),
        "xformer_skip_ff1024": lambda v: XformerSkipModel(v, embedding_dim=16,
                                                            d_model=128, nhead=4,
                                                            num_layers=2, dim_feedforward=1024,
                                                            dropout=0.1),
        "xformer_pure":        lambda v: XformerPureModel(v, embedding_dim=16,
                                                            d_model=128, nhead=4,
                                                            num_layers=2, dim_feedforward=512,
                                                            dropout=0.1),
        "xformer_skip_iha":    lambda v: XformerSkipIHAModel(v, embedding_dim=16,
                                                              d_model=128, nhead=4, num_pseudo=4,
                                                              num_layers=2, dim_feedforward=512,
                                                              dropout=0.1),
        "timeplex":            lambda v: TimePlexModel(v),
    }
    arch_names = [a.strip() for a in args.archs.split(",") if a.strip()]
    unknown = [a for a in arch_names if a not in arch_registry]
    if unknown:
        raise SystemExit(f"Unknown archs: {unknown}. Available: {sorted(arch_registry)}")
    print(f"  Running archs: {arch_names}", flush=True)

    results = {}
    for arch_name in arch_names:
        build = arch_registry[arch_name]
        print(f"\n{'='*72}\nARCH: {arch_name}\n{'='*72}", flush=True)
        set_seed(args.seed)

        vocab = Vocabulary.default()
        new_main, new_nterm = discover_tokens(combined, vocab)
        for tok, parent in new_main.items(): vocab.extend(tok, parent, fresh=False)
        for mod in sorted(new_nterm): vocab.extend_nterm_mod(mod)

        model = build(vocab).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {n_params:,}", flush=True)

        # Phase 1: pretrain from scratch on UNWARPED PSMtag
        print(f"\n--- [{arch_name}] Pretrain (UNWARPED PSMtag) ---", flush=True)
        t0 = time.time()
        rt_med, rt_mad = fit_rt_scale(pt_train["rt"].to_numpy().astype("float32"))
        train_loader = make_loader(pt_train, vocab, args.batch_size, True, rt_med, rt_mad, args.seed)
        val_loader   = make_loader(pt_val,   vocab, args.batch_size * 2, False, rt_med, rt_mad)
        test_loader  = make_loader(pt_test,  vocab, args.batch_size * 2, False, rt_med, rt_mad)
        pre_state, pre_metrics = train_loop(
            model, train_loader, val_loader,
            epochs=args.pretrain_epochs, lr=args.pretrain_lr,
            weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
            patience=args.pretrain_patience, device=device, label=f"{arch_name}/pre",
        )
        print(f"  pretrain: best val R2={pre_metrics['best_r2']:.4f} MAE={pre_metrics['best_mae']:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

        # Eval pretrain on unwarped test (baseline before warp)
        model.load_state_dict(pre_state)
        criterion = nn.HuberLoss(delta=1.0)
        _, mae_u, r2_u, d95_u = evaluate(model, test_loader, criterion, device)
        print(f"  UNWARPED test (pretrain): R2={r2_u:.4f} MAE={mae_u:.4f} d95={d95_u:.4f}", flush=True)

        # Phase 2: FT on WARPED PSMtag, restoring pretrain snapshot
        print(f"\n--- [{arch_name}] FT on WARPED PSMtag ---", flush=True)
        t1 = time.time()
        model.load_state_dict({k: v.clone() for k, v in pre_state.items()})
        wp_med, wp_mad = fit_rt_scale(pt_train_warped["rt"].to_numpy().astype("float32"))
        train_loader = make_loader(pt_train_warped, vocab, args.batch_size, True, wp_med, wp_mad, args.seed)
        val_loader   = make_loader(pt_val_warped,   vocab, args.batch_size * 2, False, wp_med, wp_mad)
        test_loader  = make_loader(pt_test_warped,  vocab, args.batch_size * 2, False, wp_med, wp_mad)
        ft_state, ft_metrics = train_loop(
            model, train_loader, val_loader,
            epochs=args.finetune_epochs, lr=args.finetune_lr,
            weight_decay=args.weight_decay, warmup_epochs=0,
            patience=args.finetune_patience, device=device, label=f"{arch_name}/ft-warped",
        )
        model.load_state_dict(ft_state)
        _, mae_w, r2_w, d95_w = evaluate(model, test_loader, criterion, device)
        print(f"  WARPED test (after FT): R2={r2_w:.4f} MAE={mae_w:.4f} d95={d95_w:.4f}  "
              f"({time.time()-t1:.1f}s)", flush=True)

        results[arch_name] = {
            "params": n_params,
            "pretrain_best_val_r2": pre_metrics["best_r2"],
            "pretrain_best_val_mae": pre_metrics["best_mae"],
            "unwarped_test": {"r2": r2_u, "mae": mae_u, "d95": d95_u},
            "warped_ft_best_val_r2": ft_metrics["best_r2"],
            "warped_ft_best_val_mae": ft_metrics["best_mae"],
            "warped_test":   {"r2": r2_w, "mae": mae_w, "d95": d95_w},
        }

    print(f"\n{'='*72}\nWARP-ROBUSTNESS SUMMARY (PSMtag pretrain -> PSMtag warped FT)\n{'='*72}", flush=True)
    print(f"{'arch':<18} {'params':>10}  {'unwarped TEST':>33}  {'warped TEST':>33}", flush=True)
    print(f"{'':<18} {'':>10}  {'R2':>10} {'MAE':>10} {'d95':>10}  {'R2':>10} {'MAE':>10} {'d95':>10}", flush=True)
    for arch_name, r in results.items():
        u, w = r["unwarped_test"], r["warped_test"]
        print(f"{arch_name:<18} {r['params']:>10,}  "
              f"{u['r2']:>10.4f} {u['mae']:>10.4f} {u['d95']:>10.4f}  "
              f"{w['r2']:>10.4f} {w['mae']:>10.4f} {w['d95']:>10.4f}", flush=True)

    out = Path("results/bench_warp_robustness.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
