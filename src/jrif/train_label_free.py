"""Train the label-free base RT model on a DIA-NN spectral library.

Thin wrapper around RTPredictor with the shipping recipe baked in
(see jrif.api.SHIPPING_RECIPE). For custom hyperparameters use the
Python API: RTPredictor.fit().
"""

import argparse
from pathlib import Path

import polars as pl

from jrif.api import RTPredictor, SHIPPING_RECIPE, SCRATCH_LR
from jrif.dataset import load_spectral_library


def _report_test(predictor: RTPredictor, test_df: pl.DataFrame) -> None:
    preds = predictor.predict(test_df, seq_col="seq")
    obs = test_df["rt"].to_numpy()
    err = abs(preds - obs)
    ss_res = ((obs - preds) ** 2).sum()
    ss_tot = ((obs - obs.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    print(f"Test (raw RT units): MAE {err.mean():.4f} | R2 {r2:.4f} | "
          f"d95 {pl.Series(err).quantile(0.95):.4f}")


def main():
    p = argparse.ArgumentParser(
        description="Train label-free base RT model (shipping recipe)")
    p.add_argument("--data-path", required=True, help="DIA-NN spectral library TSV")
    p.add_argument("--checkpoint-dir", default="checkpoints/label_free")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_df, val_df, test_df = load_spectral_library(args.data_path, seed=args.seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    predictor = RTPredictor.from_blank(arch="dilated_cnn", unified_termini=True)
    result = predictor.fit(
        train_df, seq_col="seq", rt_col="rt", val_df=val_df,
        lr=SCRATCH_LR, seed=args.seed, device=args.device,
        trained_on="label_free", **SHIPPING_RECIPE,
    )
    print(f"Best val MAE={result.best_val_mae:.4f} at epoch {result.best_epoch+1}")
    _report_test(predictor, test_df)

    # Shipping convention: checkpoints output robust-z; fit() on user data
    # adopts the user's scale.
    predictor.rt_median, predictor.rt_mad = 0.0, 1.0
    ckpt_path = Path(args.checkpoint_dir) / "dilated_cnn_v2.pt"
    predictor.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path} (z-score output)")


if __name__ == "__main__":
    main()
