# JRIF

Peptide retention time (RT) and ion mobility (IM) prediction.

```
pip install jrif
```

## Models

| name | predicts | chemistry |
|---|---|---|
| `rt_lf` | retention time | label-free |
| `rt_psmtag` | retention time | PSMtag |
| `im_psmtag` | ion mobility | PSMtag |

Weights are bundled with the package.

## Features

- Calibrate a model to your experiment using your identified peptides
- Predict RT/IM for a full spectral library, output in your experiment's units
- New modifications in your data are handled automatically
- Runs on GPU when available (CUDA or Apple), otherwise CPU

## Usage

Calibrate on your IDs, then predict your library:

```python
import pandas as pd
from jrif import RTPredictor
from jrif.checkpoints import checkpoint_path

ids = pd.read_csv("psm_ids.tsv", sep="\t")      # columns: seq, rt
lib = pd.read_csv("library.tsv", sep="\t")      # column:  seq

p = RTPredictor.from_checkpoint(checkpoint_path("rt_psmtag"))
p.calibrate(ids)
lib["pred_rt"] = p.predict(lib)
```

Ion mobility (needs a charge column):

```python
from jrif import IMPredictor

m = IMPredictor.from_checkpoint(checkpoint_path("im_psmtag"))
m.calibrate(ids)                                # columns: seq, charge, im
lib["pred_im"] = m.predict(lib)                 # columns: seq, charge
```

Sequence format: plain residues, modifications as `X[mass]` (`C[160]`,
`M[147]`, `K[308]`), N-terminal modifications as a leading `.[308]`.
UniMod notation is also accepted.

Modifications the model has never seen (e.g. `S[80]`) don't need any setup:
`calibrate()` adds new tokens to the vocabulary and learns them from your
IDs automatically.

Save and reload a calibrated model:

```python
p.save("my_experiment.pt")
p = RTPredictor.from_checkpoint("my_experiment.pt")
```

## Training new base models (maintainers)

```
uv run jrif-train-label-free --data-path <library.tsv>
uv run jrif-train-psmtag    --data-path <library.tsv | report-lib.parquet>
```
