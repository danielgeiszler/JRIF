"""Train the base IM model from a search-engine report-library parquet
(decoys dropped, Q.Value <= 0.01, (tag)/(UniMod:n) canonicalized, IM averaged
per peptide-charge pair). Shipping recipe baked in (IMPredictor.fit defaults);
custom hyperparameters go through the Python API instead.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from jrif.im import IMPredictor
from jrif.training.loaders import load_report_lib_parquet_im


def _report_test(predictor: IMPredictor, test_df: pl.DataFrame) -> None:
    preds = predictor.predict(test_df)
    obs = test_df["im"].to_numpy()
    err = np.abs(preds - obs)
    ss_res = ((obs - preds) ** 2).sum()
    ss_tot = ((obs - obs.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    print(f"Test (raw IM units): MAE {err.mean():.4f} | R2 {r2:.4f} | "
          f"d95 {np.quantile(err, 0.95):.4f}")


def main():
    p = argparse.ArgumentParser(description="Train base IM model (shipping recipe)")
    p.add_argument("--data-path", required=True, help="report-library .parquet")
    p.add_argument("--checkpoint-dir", default="checkpoints/im")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-dir", default="splits_im",
                   help="where to write the train/val/test peptide lists")
    args = p.parse_args()

    train_df, val_df, test_df = load_report_lib_parquet_im(
        args.data_path, seed=args.seed, split_dir=args.split_dir)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    predictor = IMPredictor.from_blank()
    result = predictor.fit(train_df, val_df=val_df, seed=args.seed, device=args.device)
    print(f"Best val MAE={result.best_val_mae:.4f} at epoch {result.best_epoch+1}")
    _report_test(predictor, test_df)

    # Shipping convention: checkpoints output robust-z.
    predictor.im_median, predictor.im_mad = 0.0, 1.0
    ckpt_path = Path(args.checkpoint_dir) / "dilated_cnn_im.pt"
    predictor.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path} (z-score output)")


if __name__ == "__main__":
    main()
