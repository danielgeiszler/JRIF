# JRIF — peptide retention time prediction

A transformer-on-conv ("xformer_skip") model for peptide retention time (RT) prediction. The shipped pipeline is built for one workflow: **calibrate to your LC system using identified peptides from your own experiment, then predict RT for the full search library** to improve downstream peptide-spectrum matching.

## Quick start: production workflow

Production flow is:

1. Run your search engine (FragPipe, DIA-NN, etc.) on a new experiment.
2. Filter PSMs to high confidence (e.g. 1% FDR). You now have a few thousand identified peptides with observed RTs from *your* LC gradient.
3. Fine-tune a shipped JRIF checkpoint on those identified peptides — this calibrates the predictor to your gradient.
4. Predict RT for the full spectral library you used to search (millions of peptides, only a small fraction identified). These calibrated predictions are then used by the search engine to improve RT-based filtering or rescoring on the next pass.

```python
import pandas as pd
from jrif.api import RTPredictor

# 1. Identified peptides from this experiment's PSM table (or DIA-NN report).
#    Columns: seq (peptide string), rt (observed RT from this run, native units).
ids_df = pd.read_csv("psm_ids_filtered.tsv", sep="\t")

# 2. The full spectral library you want to predict RTs for — same library you
#    searched against, including peptides that did NOT get identified.
library_df = pd.read_csv("search_library.tsv", sep="\t")     # cols: seq (rt unused)

# 3. Load the shipped base closest to your data and fine-tune on the IDs.
p = RTPredictor.from_checkpoint("checkpoints/psmtag/dilated_cnn.pt")
# Pick the (dataset, architecture) that matches your data.
# Dataset:
#   - psmtag/: trained on a PSMtag library (M[147], C[160], K[308], .[308] mods).
#   - label_free/: trained on a clean DIA-NN spectral library (no PSMtag mods).
# Architecture (per dataset):
#   - dilated_cnn.pt   ← 162k params; recommended default (faster on CPU, slightly
#                        better single-model accuracy in our benches).
#   - xformer_skip.pt  ← 569k params; transformer-on-conv variant. Use if you've
#                        validated it's better for your specific data.

p.fit(
    ids_df,
    seq_col="seq", rt_col="rt",
    epochs=150, patience=20,
    lr=1e-3,                  # bracketed optimum for fine-tuning (3e-4 < 1e-3 > 3e-3)
    batch_size=128,
    loss="mixture",           # matches the shipping recipe; handles PSM label noise
    selection_metric="mae",
    seed=42,
    device="cuda",            # or "cpu"
)

# 4. Predict RT for the full library; output is in your experiment's RT units.
library_df["pred_rt"] = p.predict(library_df, seq_col="seq")
library_df.to_csv("search_library_predRT.tsv", sep="\t", index=False)
```

The fine-tuned predictor stays calibrated to *that one experiment's* gradient. For a new experiment with a different gradient or LC system, re-load the shipped checkpoint and fine-tune again on the new experiment's IDs.

**What `fit()` does for you, automatically:**

- **Discovers new modifications** in your data. If `train_df` contains tokens the base checkpoint doesn't know (e.g. `S[80]`, `K[400]`), `fit()` extends the vocabulary and grows the model's embedding tables in-place — new rows initialized from the parent residue's row (override per-token with `fresh_mods=[...]` if the mod's chemistry diverges too far).
- **Adopts your RT scale.** The shipped checkpoints output in robust-z space (`rt_median=0`, `rt_mad=1`). After `fit()`, the predictor's RT scale is set from your train data's median/MAD, so `predict()` returns values in your library's native RT units.
- **Best-checkpoint selection on val MAE.** Holds out 10% of train as val (or pass `val_df=` explicitly); keeps the epoch with the lowest val MAE; early-stops after `patience` epochs without improvement.

## Architecture

The shipped model is `xformer_skip` (568,845 parameters):

- 5 **dilated Conv1d blocks with residual skip** (dilations 1, 2, 4, 8, 16; kernel 3; channels 28 → 64 → 64 → 128 → 128 → 128; GELU + BatchNorm; 1×1 projection on the skip path when channels change).
- Residue input per position: 12-dim physicochemical lookup (hydrophobicity, mass, pKa, vdW volume, aromaticity, H-bond donors/acceptors, pI, helix/sheet propensity, is-Lys, is-Arg) + 16-dim learned embedding = 28 channels.
- **Transformer encoder** on top of the conv stack: prepend a learnable [CLS] token, add sinusoidal positional encoding, 2 `TransformerEncoderLayer`s (d_model=128, 4 heads, dim_feedforward=512, GELU, norm_first=True). CLS output is the pooled sequence representation.
- FC head concatenates: CLS (128) + N/C-term residue embeddings (4 + 4) + N/C-term mod embeddings (4 + 4) + global features (GRAVY, net charge at pH 2.7, fraction aromatic, normalized seq length, fraction hydrophobic) = 149 dims → Linear(149, 128) → GELU → Dropout(0.2) → Linear(128, 1).

**Default vocabulary** (24 tokens): pad + 20 canonical AAs + `M[147]` (oxM) + `C[160]` (camC) + `K[308]` (K-PSMtag). N-term mod vocab: `""` (id 0) + `.[308]` (id 1). C-term mod vocab: `""` (id 0), in place for future C-term modifications. `fit()` extends these at runtime as needed.

## Training recipe (shipped)

What `train_label_free.py` and `train_psmtag.py` use to produce the shipped checkpoints:

- **Loss**: `LaplaceUniformMixture` — `(1-ε)·Laplace(0, b) + ε·Uniform(-W, W)` with learnable `b` and `ε`. Acts as a soft FDR mask: peptides with large residuals get effectively zero gradient as the posterior assigns them to the uniform contamination tail.
- **Selection metric**: validation MAE (R² is saturated above 0.997 and noisy in the third decimal).
- **Optimizer**: Muon on the dilated-conv principal weights (`lr×5`), AdamW on everything else (`lr=1e-3`, `weight_decay=1e-4`).
- **Schedule**: cosine annealing + 3-epoch linear warmup, batch size 128, max 150 epochs with patience 20.
- **Targets**: robust z-score (median/MAD on train) during training; shipped checkpoints are flattened to `(0, 1)` so `fit()` adopts the user's scale.

## Setup

```bash
uv sync
```

## CLI: train from scratch on your own library

Both training scripts default to training from scratch (no LF→PSMtag transfer). Use these only when you have a *large* labeled library (~tens of thousands of peptides) and want to produce a new shipping checkpoint. **For fine-tuning to a specific experiment's gradient, use the Python API in "Quick start" above.**

```bash
# label-free base from a DIA-NN spectral library (cols: PeptideSequence, Tr_recalibrated)
uv run jrif-train-label-free \
  --data-path data/my_lf_library.tsv \
  --epochs 150 --patience 20 \
  --checkpoint-dir checkpoints/label_free

# PSMtag base from scratch (cols: seq, rt)
uv run jrif-train-psmtag \
  --data-path data/my_psmtag_library.tsv \
  --epochs 150 --patience 20 \
  --checkpoint-dir checkpoints/psmtag
```

The `--base-checkpoint` arg on `train_psmtag.py` is optional and only used to warm-start the new base from a previous shipping checkpoint; it's not for fine-tuning to a new experiment. For arbitrary DataFrames, use `RTPredictor.fit()` (see "Quick start").

## CLI: predict on a library TSV

```bash
# Use the psmtag checkpoint for libraries with PSMtag modifications
uv run jrif-predict \
  --input-tsv my_library.tsv \
  --output-tsv my_library_predRT.tsv \
  --checkpoint checkpoints/psmtag/dilated_cnn.pt

# label-free libraries
uv run jrif-predict \
  --input-tsv my_library.tsv \
  --output-tsv my_library_predRT.tsv \
  --checkpoint checkpoints/label_free/dilated_cnn.pt

# When PSMtag is applied but not written into sequences, --add-psmtag annotates
# every N-terminus and K with the tag before tokenization
uv run jrif-predict \
  --input-tsv my_psmtag_library.tsv \
  --output-tsv my_psmtag_library_predRT.tsv \
  --checkpoint checkpoints/psmtag/dilated_cnn.pt \
  --add-psmtag
```

> **Note:** `jrif-predict` outputs RT predictions in the shipped checkpoint's
> z-space (`rt_median=0`, `rt_mad=1`). To get RT in physical units, fine-tune
> on your experiment's PSM IDs first via the Python API (see "Quick start"
> above) — that adjusts the RT scale to your gradient and shifts the output
> back into native units automatically.

## Tests

```bash
uv sync --extra dev
uv run pytest
```

## Repository layout

```
JRIF/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── src/jrif/
│   ├── __init__.py
│   ├── api.py              # RTPredictor — public DataFrame API
│   ├── dataset.py          # Vocabulary, tokenizers, RTDataset, library loaders
│   ├── loss.py             # LaplaceUniformMixture + NormalNormalMixture
│   ├── model.py            # RTModel (dilated CNN; legacy registry entry)
│   ├── model_xformer.py    # XformerSkipModel — the shipped architecture
│   ├── muon.py             # Muon optimizer (used for conv weights)
│   ├── io.py               # TSV format detection + PSMtag annotation helper
│   ├── train_label_free.py # CLI: train label-free base from scratch
│   ├── train_psmtag.py     # CLI: train PSMtag base (from scratch or from base)
│   └── predict_library.py  # CLI: predict RT for a library TSV
├── tests/                  # Tokenizer, vocab-extension, checkpoint, API smoke tests
├── data/                   # Place training libraries here
└── checkpoints/            # Trained checkpoints — one file per (dataset, arch)
    ├── label_free/
    │   ├── dilated_cnn.pt   # 162k params, default recommended
    │   └── xformer_skip.pt  # 569k params, transformer-on-conv variant
    └── psmtag/
        ├── dilated_cnn.pt
        └── xformer_skip.pt
```
