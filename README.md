# JRIF — retention time prediction

A dilated-CNN model for peptide retention time (RT) prediction. Designed to:

1. Train a **label-free base model** from a DIA-NN spectral library (e.g. human HY library).
2. Fine-tune that model on a **modified-peptide library** (PSMtag: `.[308.1161]` N-terminal tag + `K[308.1161]` side-chain tag).
3. Use either checkpoint to **predict RT** for arbitrary peptide lists in FragPipe/DIA-NN spectral-library TSV format.

## Architecture

- 5 dilated `Conv1d` blocks (dilations 1, 2, 4, 8, 16; kernel 3; GELU + BatchNorm).
- Residue embedding = fixed physicochemical features (12 dims) + learned embedding (16 dims) = 28 input channels.
- Masked global average pool over the conv stack.
- FC head concatenates: GAP output, N/C-term AA embeddings, N/C-term mod embeddings, and handcrafted global features (GRAVY, net charge at pH 2.7, fraction aromatic, seq len, fraction hydrophobic).

Vocab (24 tokens): pad + 20 canonical AAs + `M[147]` (oxM) + `C[160]` (camC) + `K[308]` (K-PSMtag).
N-term mod embedding distinguishes "no mod" (id=0) from `.[308.1161]` (id=1). C-term mod embedding is in place for future α-carboxyl mods (currently always id=0).

## Setup

```bash
uv sync
```

## Training

**Step 1**: train the label-free base model on a DIA-NN library.

```bash
uv run jrif-train-label-free \
  --data-path data/DIANNv1p9_JD0507_509_LF_HY.tsv \
  --checkpoint-dir checkpoints/label_free \
  --epochs 150
```

Expected val R² around 0.997 after convergence. Produces `checkpoints/label_free/best.pt`.

**Step 2**: fine-tune on the PSMtag library, starting from the label-free base checkpoint.

```bash
uv run jrif-train-psmtag \
  --data-path data/psmtag_library.tsv \
  --base-checkpoint checkpoints/label_free/best.pt \
  --checkpoint-dir checkpoints/psmtag \
  --epochs 50
```

Expected test R² around 0.99 with ~35K PSMtag peptides. Produces `checkpoints/psmtag/best.pt`.

## Prediction

Replace the RT column of a FragPipe-style library TSV with our predictions.

```bash
uv run jrif-predict \
  --input-tsv my_library.tsv \
  --output-tsv my_library_predRT.tsv \
  --checkpoint checkpoints/psmtag/best.pt
```

For libraries **without PSMtag modifications** (standard label-free), use the label-free checkpoint:

```bash
uv run jrif-predict \
  --input-tsv my_library.tsv \
  --output-tsv my_library_predRT.tsv \
  --checkpoint checkpoints/label_free/best.pt
```

For libraries **with PSMtag applied** (where the tags are implicit and not written in sequences), add `--add-psmtag` so the script annotates every N-terminus and K with the tag before tokenization:

```bash
uv run jrif-predict \
  --input-tsv my_psmtag_library.tsv \
  --output-tsv my_psmtag_library_predRT.tsv \
  --checkpoint checkpoints/psmtag/best.pt \
  --add-psmtag
```

## Python API

For programmatic use, hand a pandas or polars DataFrame of `{seq, rt}` to
`RTPredictor.fit(...)`. New modification tokens encountered in the input are
detected automatically and the embedding tables grow to match (inheriting from
the parent residue by default). The trained predictor saves a self-describing
checkpoint that ships its own vocabulary and RT scale, so it round-trips
through `from_checkpoint(...).predict(df)` without any external configuration.

```python
import polars as pl
from jrif import RTPredictor

train_df = pl.read_csv("my_train.tsv", separator="\t")  # cols: seq, rt
val_df   = pl.read_csv("my_val.tsv",   separator="\t")

predictor = RTPredictor.from_checkpoint("checkpoints/label_free/best.pt")
result = predictor.fit(
    train_df, val_df=val_df,
    epochs=50, lr=1e-4, batch_size=128,
    fresh_mods=["S[80]"],   # opt-in: don't inherit physico features from S
)
print(result.best_val_r2, result.best_val_mae)

predictor.save("checkpoints/my_finetune/best.pt")

# Later, in another process:
loaded = RTPredictor.from_checkpoint("checkpoints/my_finetune/best.pt")
preds = loaded.predict(some_df)   # numpy array of RTs in original units
```

If you don't pass `val_df`, `fit` holds out a peptide-level slice of `train_df`
controlled by `val_fraction` (default 0.1). Training metrics are returned as
a `FitResult`, and TensorBoard logging is opt-in via `log_dir=...`.

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
│   ├── api.py             # RTPredictor — public DataFrame API
│   ├── dataset.py         # Vocabulary, tokenizers, RTDataset, library loaders
│   ├── model.py           # RTModel (dilated CNN, runtime-extensible vocab)
│   ├── muon.py            # Muon optimizer (used for conv weights)
│   ├── io.py              # TSV format detection + PSMtag annotation helper
│   ├── train_label_free.py # CLI: train label-free base model on DIA-NN library
│   ├── train_psmtag.py     # CLI: fine-tune base on PSMtag library
│   └── predict_library.py  # CLI: predict RT for a library TSV
├── tests/                 # Tokenizer, vocab-extension, checkpoint, API smoke tests
├── data/                  # Place training libraries here
└── checkpoints/           # Trained checkpoints land here
```
