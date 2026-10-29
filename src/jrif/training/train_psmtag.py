"""Train the PSMtag base RT model.

Accepts either a FragPipe PSMtag library TSV or a search-engine report-library parquet
(decoys dropped, Q.Value <= 0.01, (tag)/(UniMod:n) canonicalized). Shipping
recipe baked in (see jrif.api.SHIPPING_RECIPE); custom hyperparameters go
through the Python API instead.
"""

import argparse
from pathlib import Path

from jrif.api import RTPredictor, SHIPPING_RECIPE, SCRATCH_LR, FINETUNE_LR
from jrif.training.loaders import load_psmtag_library, load_report_lib_parquet
from jrif.training.train_label_free import _report_test


def main():
    p = argparse.ArgumentParser(
        description="Train PSMtag base RT model (shipping recipe)")
    p.add_argument("--data-path", required=True,
                   help="PSMtag library TSV or report-library .parquet")
    p.add_argument("--base-checkpoint", default=None,
                   help="Optional: warm-start from an existing checkpoint "
                        "(fine-tune lr). Default: train from scratch.")
    p.add_argument("--checkpoint-dir", default="checkpoints/psmtag")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-dir", default="splits_psmtag",
                   help="where to write the train/val/test peptide lists")
    args = p.parse_args()

    if str(args.data_path).endswith(".parquet"):
        train_df, val_df, test_df = load_report_lib_parquet(
            args.data_path, seed=args.seed, split_dir=args.split_dir)
    else:
        train_df, val_df, test_df = load_psmtag_library(
            args.data_path, seed=args.seed, split_dir=args.split_dir)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if args.base_checkpoint:
        print(f"Warm-starting from {args.base_checkpoint}")
        predictor = RTPredictor.from_checkpoint(args.base_checkpoint)
        lr = FINETUNE_LR
    else:
        predictor = RTPredictor.from_blank()
        lr = SCRATCH_LR

    result = predictor.fit(
        train_df, seq_col="seq", rt_col="rt", val_df=val_df,
        lr=lr, seed=args.seed, device=args.device,
        trained_on="psmtag", **SHIPPING_RECIPE,
    )
    print(f"Best val MAE={result.best_val_mae:.4f} at epoch {result.best_epoch+1}")
    _report_test(predictor, test_df)

    predictor.rt_median, predictor.rt_mad = 0.0, 1.0
    ckpt_path = Path(args.checkpoint_dir) / "dilated_cnn_v2.pt"
    predictor.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path} (z-score output)")


if __name__ == "__main__":
    main()
