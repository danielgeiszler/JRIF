# JRIF — peptide retention time & ion mobility prediction

[![PyPI](https://img.shields.io/pypi/v/jrif.svg)](https://pypi.org/project/jrif/)

Install: `pip install jrif`

A compact dilated-CNN library for predicting peptide retention time (RT) and
ion mobility (IM). The production
workflow: **ship a pretrained base per property, fine-tune it to each
experiment using that experiment's identified peptides, then predict the full
search library.**

Two model families, one backbone (162k parameters, unified-termini dilated
CNN). Currently supported modalities:

- **Label-free**
  - **RT** — `RTPredictor`, input: peptide sequence —
    `checkpoint_path("rt_lf")`
- **PSMtag**
  - **RT** — `RTPredictor`, input: peptide sequence —
    `checkpoint_path("rt_psmtag")`
  - **IM** — `IMPredictor`, input: sequence + precursor charge (z 2-4) —
    `checkpoint_path("im_psmtag")`

There is no label-free IM base yet; other chemistries are reachable from the
nearest base via `calibrate()` (novel mods are added to the vocabulary
automatically, at some accuracy cost vs a matched base).

Shipped checkpoints output robust-z scores (`rt_median=0, rt_mad=1`);
calibration/fine-tuning adopts the user's units automatically.

## Setup

```bash
uv sync
```

Device selection is automatic everywhere (`cuda` → `mps` → `cpu`); pass
`device=` to override.

## Calibrating to an experiment (the main user path)

`calibrate()` adapts a shipped base to one experiment's identified peptides
and picks the right strategy empirically:

```python
import pandas as pd
from jrif import RTPredictor

# identified peptides from this experiment (1% FDR), observed RTs in native units
ids_df = pd.read_csv("psm_ids.tsv", sep="\t")        # cols: seq, rt

from jrif.checkpoints import checkpoint_path
p = RTPredictor.from_checkpoint(checkpoint_path("rt_psmtag"))
report = p.calibrate(ids_df)                          # seq_col=/rt_col= to rename
print(report)   # {'mode': 'warp+finetune', 'holdout_rank': 0.99, 'holdout_mae': ...}

# predict the full search library, output in the experiment's native RT units
library = pd.read_csv("search_library.tsv", sep="\t") # col: seq
library["pred_rt"] = p.predict(library)

p.save("calibrated_rt.pt")   # the warp is stored in the checkpoint
```

What `calibrate()` does internally:

1. **Rank check.** Holds out 20% of the IDs and checks whether the frozen
   base can rank them (Spearman ≥ 0.95).
2. **Warp-first path** (normal case — matched chemistry, new gradient): fits
   a monotone isotonic warp from model output to observed RT, fine-tunes on
   *inverse-warped* targets (training happens in the model's own space, so
   gradients go to chemistry, never to re-learning the gradient shape),
   refits the warp, and keeps warp-only or warp+finetune — whichever scores
   better on the holdout. Benchmarks: ~1.2% of gradient span from 300 IDs,
   ~0.9% from 2,000.
3. **Fallback** (`finetune_raw`): if the rank check fails — the base has
   never seen the dominant chemistry (e.g. a fixed-mod experiment on the
   label-free base) — it fine-tunes on raw targets instead. This regime
   needs more IDs; prefer the matched base.

**Novel modifications**: unknown mod tokens in the IDs are discovered and
added to the vocabulary automatically (single-phase). `calibrate(df,
curriculum=True)` instead stages it — calibrate on vocabulary-complete
peptides first, then extend the vocabulary and continue on the full set.

Pick the base whose training chemistry matches the data: `psmtag/` for
PSMtag-tagged libraries (`.[308]`, `K[308]`, `C[160]`, `M[147]`), `label_free/`
for unmodified/camC libraries.

**Guidance on ID counts** (test MAE as % of gradient span, matched base):
~1.6% at 100 IDs, 1.2% at 300, 0.9% at 2k, 0.8% at 20k, 0.65% at 78k — no
plateau, so feed it every confident ID available.

## Prediction inputs

`predict()` takes a pandas or polars DataFrame and returns a numpy array,
row-aligned, duplicates allowed.

**Columns.** RT: `seq` (override with `seq_col=`). IM: `seq` + `charge`
(integer precursor charge; `charge_col=`).

**Sequence grammar.** Plain residues (`ACDEFK...`); modified residues as
`X[mass]` with integer mass (`M[147]`, `C[160]`, `K[308]`, `S[80]`);
N-terminal mods as a leading `.[mass]` (`.[308]PEPTIDEK[308]`). UniMod
notation (`M(UniMod:35)`, `C(UniMod:4)`) is auto-detected and canonicalized.
In non-UniMod formats a bare `C` is treated as `C[160]` (carbamidomethyl
assumed). Max 50 tokens; longer sequences are truncated. A modified residue
not in the vocabulary falls back to its bare parent rather than erroring.

**Units.** A shipped base returns robust-z scores; after `calibrate()`/
`fit()` on an experiment's IDs, predictions are in that experiment's native
units (affine scale and isotonic warp composed automatically).

**Performance.** Single process, batch-parallel on the selected device
(default batch 2048, `batch_size=`); `device=` on every predict/fit/
calibrate call, auto-resolved `cuda` → `mps` → `cpu` when omitted. Throughput
~60k peptides/s on Apple GPU, ~29k/s on CPU (multicore BLAS) — a 5M-peptide
library in ~2-3 minutes. No multi-GPU sharding.

## Ion mobility

```python
from jrif import IMPredictor

ids = pd.DataFrame({"seq": [...], "charge": [...], "im": [...]})
m = IMPredictor.from_checkpoint(checkpoint_path("im_psmtag"))
m.calibrate(ids)                             # same warp-first design as RT
library["pred_im"] = m.predict(library)      # library needs seq + charge cols
```

Charge enters as a per-position input channel — benchmarked against a
head-level feature and no-charge: the input channel is the only variant that
fixes higher charge states (z=4 MAE 0.020 vs 0.090). `IMPredictor` has the
same `fit/predict/save/from_checkpoint` surface as RT; checkpoints are tagged
`kind: "im"` and the two loaders refuse each other's files.

## Fine-tuning directly (bypassing calibrate)

`fit()` is the raw training entry point; `calibrate()` is preferred for
experiment adaptation, `fit()` for custom workflows:

```python
p = RTPredictor.from_checkpoint(checkpoint_path("rt_psmtag"))
p.fit(df, seq_col="seq", rt_col="rt",
      epochs=150, patience=20, lr=1e-3, batch_size=128,
      loss="mixture", selection_metric="mae", seed=42)
```

- **New modifications are discovered automatically**: unknown tokens (e.g.
  `S[80]`) extend the vocabulary in place, initialized from the parent
  residue's embedding (pass `fresh_mods=[...]` for chemistry too far from the
  parent — the PSMtag tag is the precedent).
- **lr 1e-3 for both scratch and fine-tune** (bracketed optimum: 3e-4 < 1e-3 > 3e-3).
- **`loss="mixture"`** (Laplace + uniform contamination, learnable scale and
  outlier weight) is the shipping default — it soft-masks mislabeled PSMs.
  Confirmed against normal/normal and HL-Gauss alternatives.
- **`batch_size="auto"`** picks a throughput-optimal batch and rescales the
  epoch/patience budget to keep total optimizer steps constant (quality is
  step-budget-bound, not batch-bound). Use for fast iteration; the default
  128 for shipping-quality runs.
- Any `fit()` clears a stored calibration warp (the weights moved).

## Training base models from scratch

Use the CLIs when producing new shipping checkpoints from a large library
(tens of thousands of peptides). The full recipe (mixture loss, val-MAE
selection, Muon+AdamW, cosine schedule + warmup, batch 128, 150 epochs,
patience 20) is baked in — see `SHIPPING_RECIPE` in `jrif/api.py`.

```bash
# label-free RT base from a spectral library TSV
# (cols: PeptideSequence [, ModifiedPeptide|ModifiedSequence], Tr_recalibrated)
uv run jrif-train-label-free --data-path data/JD_LF.tsv --device mps

# PSMtag RT base from a library TSV or a search-engine report-library parquet
# (parquet path drops decoys, filters Q.Value <= 0.01, canonicalizes mods)
uv run jrif-train-psmtag --data-path data/report-lib.parquet --device mps
```

Both write `<checkpoint-dir>/dilated_cnn_v2.pt` with the RT scale flattened
to z (shipping convention). `--base-checkpoint` on `jrif-train-psmtag`
warm-starts from an existing checkpoint at the fine-tune recipe.

The IM base is trained through the Python API (no CLI yet): build the
`(seq, charge, im)` frame, `IMPredictor.from_blank().fit(...)`, set
`im_median, im_mad = 0.0, 1.0`, and `save()`.

## Batch prediction CLI

```bash
uv run jrif-predict \
  --input-tsv my_library.tsv --output-tsv my_library_predRT.tsv \
  --checkpoint src/jrif/checkpoints/rt_psmtag.pt \
  [--add-psmtag]   # annotate N-termini and K with the tag before tokenization
```

Output is in the checkpoint's units — z-scores for shipped bases; calibrate
first (Python API) for native-unit predictions.

## Architecture (both properties)

- Tokens: 20 AAs + modified residues (`X[mass]`) + terminal mods as in-sequence
  tokens (`.[308]` at position 0) — one vocabulary, one place (format_version-2
  "unified termini" checkpoints). The vocabulary lives in the checkpoint and
  grows at runtime via `extend_vocab`.
- Per position: 12 physicochemical features + 16-dim learned embedding
  (+ charge channel for IM) → 5 dilated conv blocks (dil 1/2/4/8/16, GELU+BN)
  → masked mean pool → small FC head with global features.
- Conv forward uses a single-GEMM decomposition on CPU (PyTorch's dilated-conv
  CPU kernel is pathologically slow) and folds BatchNorm at eval — CPU
  training ~5x and inference ~11x faster than the naive path.
- Training: Muon on conv weights (lr×5), AdamW elsewhere; robust-z targets.

## Repository layout

```
src/jrif/                # runtime package — everything a consumer needs
├── __init__.py          # public API: RTPredictor, IMPredictor
├── api.py               # RTPredictor: fit / calibrate / predict / save, SHIPPING_RECIPE
├── im.py                # IMModel / IMDataset / IMPredictor
├── model.py             # RTModel, extend_vocab, fast conv paths
├── dataset.py           # Vocabulary, tokenizers, RTDataset
├── loss.py              # LaplaceUniformMixture (+ alternatives)
├── muon.py              # Muon optimizer
├── io.py                # library-TSV format detection, PSMtag annotation
├── predict_library.py   # CLI: batch prediction
└── training/            # base-model building (internal; never runs on user machines)
    ├── loaders.py       # spectral-library / PSMtag / report-lib ingestion + splits
    ├── train_label_free.py
    └── train_psmtag.py
tests/                  # tokenizer, vocab-extension, calibrate, IM, checkpoint tests
src/jrif/checkpoints/   # shipped weights, bundled into the wheel
checkpoints/            # local-only training outputs (gitignored)
data/                   # training libraries (gitignored; no LFS)
```

Benchmark scripts and experiment results were pruned from the working tree
for production; they are preserved in git history (commit `ee20581` and
earlier).

## Tests

```bash
uv sync --extra dev
uv run pytest
```
