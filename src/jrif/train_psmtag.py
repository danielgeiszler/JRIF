"""Fine-tune the base library model on a PSMtag spectral library.

Full fine-tune (all params trainable). Loads from --base-checkpoint and saves
the best-val-R2 checkpoint (including rt_median / rt_mad so the predictor can
back-transform z-scored outputs to iRT)."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from jrif.dataset import RTDataset, load_psmtag_library
from jrif.model import RTModel, load_state_dict_compat
from jrif.muon import Muon


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, tgts = [], []
    with torch.no_grad():
        for seqs, nterm, cterm, targets in loader:
            seqs, nterm, cterm, targets = (
                seqs.to(device), nterm.to(device), cterm.to(device), targets.to(device),
            )
            y = model(seqs, nterm, cterm)
            total_loss += criterion(y, targets).item() * len(targets)
            preds.append(y.cpu()); tgts.append(targets.cpu())
    p = torch.cat(preds); t = torch.cat(tgts)
    n = len(t)
    err = (p - t).abs()
    ss_res = ((t - p) ** 2).sum().item(); ss_tot = ((t - t.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return total_loss / n, err.mean().item(), r2, err.quantile(0.95).item()


def main():
    p = argparse.ArgumentParser(description="Fine-tune base RT model on PSMtag library")
    p.add_argument("--data-path", required=True, help="PSMtag library TSV")
    p.add_argument("--base-checkpoint", required=True, help="Base library checkpoint")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", default="runs/psmtag")
    p.add_argument("--checkpoint-dir", default="checkpoints/psmtag")
    p.add_argument("--exclude-oxm", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    print("Loading PSMtag library...")
    train_df, val_df, test_df = load_psmtag_library(
        args.data_path, seed=args.seed, exclude_oxm=args.exclude_oxm,
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_ds = RTDataset(train_df)
    rt_med, rt_mad = train_ds.rt_median, train_ds.rt_mad
    print(f"RT normalization: median={rt_med:.2f}, MAD={rt_mad:.2f}")
    val_ds = RTDataset(val_df, rt_median=rt_med, rt_mad=rt_mad)
    test_ds = RTDataset(test_df, rt_median=rt_med, rt_mad=rt_mad)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    model = RTModel().to(device)
    print(f"Loading base checkpoint: {args.base_checkpoint}")
    ckpt = torch.load(args.base_checkpoint, weights_only=True)
    load_state_dict_compat(model, ckpt["model_state_dict"])

    muon_params, adamw_params = [], []
    for name, pr in model.named_parameters():
        if "convs." in name and "conv.weight" in name and pr.ndim >= 2:
            muon_params.append(pr)
        else:
            adamw_params.append(pr)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=args.lr, weight_decay=args.weight_decay)
    opt_muon = Muon(muon_params, lr=args.lr * 5, momentum=0.95)
    sched_adamw = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adamw, T_max=args.epochs)
    sched_muon = torch.optim.lr_scheduler.CosineAnnealingLR(opt_muon, T_max=args.epochs)
    criterion = nn.HuberLoss(delta=1.0)

    writer = SummaryWriter(log_dir=args.log_dir)
    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_r2 = -float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0; n = 0
        for seqs, nterm, cterm, targets in train_loader:
            seqs, nterm, cterm, targets = (
                seqs.to(device), nterm.to(device), cterm.to(device), targets.to(device),
            )
            loss = criterion(model(seqs, nterm, cterm), targets)
            opt_adamw.zero_grad(); opt_muon.zero_grad()
            loss.backward()
            opt_adamw.step(); opt_muon.step()
            running_loss += loss.item() * len(targets); n += len(targets)
        sched_adamw.step(); sched_muon.step()

        val_loss, val_mae, val_r2, val_d95 = evaluate(model, val_loader, criterion, device)
        lr_now = opt_adamw.param_groups[0]["lr"]
        writer.add_scalar("Loss/train", running_loss / n, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/val_MAE", val_mae, epoch)
        writer.add_scalar("Metrics/val_R2", val_r2, epoch)
        writer.add_scalar("Metrics/val_d95", val_d95, epoch)
        writer.add_scalar("LR", lr_now, epoch)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | train_loss {running_loss/n:.4f} | "
            f"val_loss {val_loss:.4f} | val_MAE {val_mae:.4f} | val_R2 {val_r2:.4f} | "
            f"val_d95 {val_d95:.4f} | LR {lr_now:.2e}"
        )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_r2": val_r2, "val_mae": val_mae, "val_loss": val_loss,
                    "rt_median": rt_med, "rt_mad": rt_mad,
                    "args": vars(args),
                },
                ckpt_dir / "best.pt",
            )
            print(f"  -> saved best (val_r2={val_r2:.4f})")

    writer.close()

    print("\n--- Test evaluation ---")
    ckpt = torch.load(ckpt_dir / "best.pt", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_mae, test_r2, test_d95 = evaluate(model, test_loader, criterion, device)
    print(f"Test: loss {test_loss:.4f} | MAE {test_mae:.4f} | R2 {test_r2:.4f} | d95 {test_d95:.4f}")


if __name__ == "__main__":
    main()
