"""Fine-tune the base library model on a PSMtag spectral library.

Loads a base checkpoint, discovers any new modification tokens in the PSMtag
data (PSMtag's K[308] and N-term .[308] tag are usually already in the default
vocab; novel mods are picked up automatically), and full-fine-tunes. Saves a
self-describing checkpoint.
"""

import argparse
from pathlib import Path

import polars as pl

from jrif.api import RTPredictor, _ARCH_BUILDERS
from jrif.dataset import load_psmtag_library


def main():
    p = argparse.ArgumentParser(description="Fine-tune base RT model on PSMtag library")
    p.add_argument("--data-path", required=True, help="PSMtag library TSV")
    p.add_argument("--base-checkpoint", default=None,
                   help="Optional: warm-start from an existing checkpoint. "
                        "When omitted, train from scratch (recommended — this matches the "
                        "from-scratch PSMtag protocol used in our benchmarks).")
    p.add_argument("--arch", default="xformer_skip", choices=sorted(_ARCH_BUILDERS),
                   help="Model architecture (used only when training from scratch)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", default="runs/psmtag")
    p.add_argument("--checkpoint-dir", default="checkpoints/psmtag")
    p.add_argument("--exclude-oxm", action="store_true")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stop if val MAE hasn't improved for this many epochs")
    p.add_argument("--loss", default="mixture", choices=["mae", "huber", "mixture"],
                   help="Training loss. mixture (Laplace+Uniform) is recommended for "
                        "PSMtag-style prefiltered-but-noisy data.")
    p.add_argument("--selection-metric", default="mae", choices=["mae", "r2"],
                   help="Metric used for best-checkpoint selection.")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    print("Loading PSMtag library...")
    train_df, val_df, test_df = load_psmtag_library(
        args.data_path, seed=args.seed, exclude_oxm=args.exclude_oxm,
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print(f"Loss: {args.loss}  selection: val_{args.selection_metric.upper()}")
    if args.base_checkpoint:
        print(f"Loading base checkpoint: {args.base_checkpoint}")
        predictor = RTPredictor.from_checkpoint(args.base_checkpoint)
    else:
        print(f"Training from scratch (arch={args.arch})")
        predictor = RTPredictor.from_blank(arch=args.arch)
    result = predictor.fit(
        train_df,
        seq_col="seq", rt_col="rt",
        val_df=val_df,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        patience=args.patience,
        trained_on="psmtag",
        loss=args.loss,
        selection_metric=args.selection_metric,
    )
    print(f"Best val R2={result.best_val_r2:.4f} MAE={result.best_val_mae:.4f} at epoch {result.best_epoch+1}")

    ckpt_path = Path(args.checkpoint_dir) / "best.pt"
    predictor.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    preds = predictor.predict(test_df, seq_col="seq")
    obs = test_df["rt"].to_numpy()
    err = abs(preds - obs)
    ss_res = ((obs - preds) ** 2).sum()
    ss_tot = ((obs - obs.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    print(
        f"Test (raw RT units): MAE {err.mean():.4f} | R2 {r2:.4f} | "
        f"d95 {pl.Series(err).quantile(0.95):.4f}"
    )


if __name__ == "__main__":
    main()
