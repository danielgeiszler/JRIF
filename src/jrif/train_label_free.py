"""Train the label-free base RT model on a DIA-NN spectral library.

Produces a label-free iRT predictor with a self-describing checkpoint
(carries vocab + rt_median/rt_mad). Thin wrapper around RTPredictor.
"""

import argparse
from pathlib import Path

import polars as pl

from jrif.api import RTPredictor
from jrif.dataset import load_spectral_library


def main():
    p = argparse.ArgumentParser(
        description="Train label-free base RT model on a DIA-NN spectral library"
    )
    p.add_argument("--data-path", required=True, help="DIA-NN spectral library TSV")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-dir", default="runs/label_free")
    p.add_argument("--checkpoint-dir", default="checkpoints/label_free")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stop if val R2 hasn't improved for this many epochs")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    print("Loading library...")
    train_df, val_df, test_df = load_spectral_library(args.data_path, seed=args.seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    predictor = RTPredictor.from_blank()
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
        trained_on="label_free",
    )
    print(f"Best val R2={result.best_val_r2:.4f} MAE={result.best_val_mae:.4f} at epoch {result.best_epoch+1}")

    ckpt_path = Path(args.checkpoint_dir) / "best.pt"
    predictor.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Final test eval (z-space metrics; comparable to legacy script)
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
