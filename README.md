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

**Step 1**: train the base model on the DIA-NN library.

```bash
uv run jrif-train-library \
  --data-path data/DIANNv1p9_JD0507_509_LF_HY.tsv \
  --checkpoint-dir checkpoints/library \
  --epochs 150
```

Expected val R² around 0.997 after convergence. Produces `checkpoints/library/best.pt`.

**Step 2**: fine-tune on the PSMtag library, starting from the base checkpoint.

```bash
uv run jrif-train-psmtag \
  --data-path data/psmtag_library.tsv \
  --base-checkpoint checkpoints/library/best.pt \
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

For libraries **without PSMtag modifications** (standard label-free), use the library checkpoint:

```bash
uv run jrif-predict \
  --input-tsv my_library.tsv \
  --output-tsv my_library_predRT.tsv \
  --checkpoint checkpoints/library/best.pt
```

For libraries **with PSMtag applied** (where the tags are implicit and not written in sequences), add `--add-psmtag` so the script annotates every N-terminus and K with the tag before tokenization:

```bash
uv run jrif-predict \
  --input-tsv my_psmtag_library.tsv \
  --output-tsv my_psmtag_library_predRT.tsv \
  --checkpoint checkpoints/psmtag/best.pt \
  --add-psmtag
```

## Repository layout

```
JRIF/
├── pyproject.toml
├── README.md
├── src/jrif/
│   ├── __init__.py
│   ├── dataset.py        # Tokenizers, VOCAB, RTDataset, library loaders
│   ├── model.py          # RTModel (dilated CNN)
│   ├── muon.py           # Muon optimizer (used for conv weights)
│   ├── train_library.py  # Train base model on DIA-NN library
│   ├── train_psmtag.py   # Fine-tune base on PSMtag library
│   └── predict_library.py  # Predict RT for a library TSV
├── data/                 # Place training libraries here
└── checkpoints/          # Trained checkpoints land here
```
