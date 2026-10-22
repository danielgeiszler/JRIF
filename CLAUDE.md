# JRIF — agent guardrails

## Architecture in one paragraph

`RTPredictor` (`src/jrif/api.py`) is the public DataFrame API. It owns one
`RTModel` (a dilated CNN) plus a `Vocabulary` (token-id maps + lineage) plus
a robust z-score scale `(rt_median, rt_mad)`. The model's embedding tables
and physico/global lookup buffers are sized from the Vocabulary at build
time; `RTModel.extend_vocab(...)` grows them in-place when fine-tune data
contains modification tokens the base model has not seen. New rows inherit
from the parent residue's row by default; tokens listed in `fresh_mods=...`
opt out of inheritance and start at zero (the K[308] precedent).

## Hard rules

- **Never delete files without explicit permission.** This includes `.legacy`
  backups, old checkpoints, generated artifacts, anything. "Auto mode" /
  "minimize interruptions" does NOT cover destructive operations. If a file
  looks redundant or stale, leave it alone and ask. Same goes for `rm -rf`,
  overwriting non-version-controlled artifacts, dropping checkpoints during
  refactors, or any other action where the file does not come back from
  `git checkout`.


- **The vocabulary lives in the checkpoint.** Never hardcode `VOCAB_SIZE` or
  `NTERM_MOD_VOCAB_SIZE` outside `Vocabulary.default()`. Anything that needs
  vocab size reads it from `predictor.vocab` / `model.vocab.size`.

- **New modification tokens go through `extend_vocab`.** Don't construct an
  `RTModel` with an arbitrary `vocab_size`; build with a Vocabulary and
  call `extend_vocab(...)` to grow. Embedding rows for new tokens MUST
  inherit from the parent AA (or be explicitly `fresh`) so that predictions
  on peptides without the new token are unchanged. There's a regression
  test for this in `tests/test_vocab_extend.py`.

- **Optimizers must be built after `extend_vocab`.** `extend_vocab` rebinds
  `self.embedding` etc. to fresh `nn.Embedding` modules; any optimizer
  constructed before that holds references to the old (now-detached)
  parameters. `RTPredictor.fit` already runs `extend_vocab` before
  `_make_optimizers` — preserve that order.

- **All internal computation is in robust-z space.** `RTPredictor.fit`
  computes `(rt_median, rt_mad)` from the training split via
  `dataset.fit_rt_scale`, z-scores the targets, and trains/evaluates
  entirely in z. The model only ever sees z. After `fit`, the predictor
  remembers that scale; `predict()` back-transforms from z to the
  fit-dataset's original RT units (`pred_z * rt_mad + rt_median`).

- **Shipped pretrained checkpoints output z-scores, not iRT.** The
  `checkpoints/label_free/best.pt` and `checkpoints/psmtag/best.pt`
  artifacts ship with `rt_median=0` and `rt_mad=1`, so their back-transform
  is the identity. The original training datasets had iRT-like scales, but
  those scales are arbitrary to downstream callers — z-score output is the
  canonical contract for these checkpoints. Don't restore the arbitrary
  iRT scale onto a shipped checkpoint. After a user calls `.fit(my_df)` on
  their own data, the predictor adopts `my_df`'s scale and `predict`
  returns numbers back in those units.

- **Checkpoints carry their own `format_version`.** `from_checkpoint`
  branches on this: format-versions 1 and 2 read the embedded vocab + RT
  scale; missing version is treated as legacy and falls back to
  `Vocabulary.default()` plus `load_legacy_state_dict` (slices oversized
  legacy `nterm_mod_embedding` rows). Don't bypass either path.

- **Unified-termini mode (format_version 2).** The preferred representation
  for new models: `RTPredictor.from_blank(unified_termini=True)` makes
  terminal mods main-vocab tokens emitted in the sequence (`.[308]` at
  position 0) with NO terminal head features (head is 133-dim, not 149).
  Validated to tie the legacy architecture on PSMtag data
  (`results/bench_termmods_pooling.json`, `inseq_clean` variant). The
  separate nterm/cterm channels and head injections remain only for loading
  v1/legacy checkpoints — the shipped checkpoints are all v1 and their
  training data is not on this machine, so the legacy path must keep
  working. `unified_termini` is dilated_cnn-only for now (xformer_skip
  raises).

- **Internal data shape is polars.** `RTDataset` and
  `RTPredictor` accept either pandas or polars at the boundary and convert
  via `dataset.to_polars`. New code should not introduce pandas dependencies
  in hot paths.

## Test discipline

`tests/test_vocab_extend.py` is the most important test file: any change to
`RTModel.extend_vocab`, the physico/global lookup builders, or how new
embedding rows are initialized must keep these tests green. They check that
extension does NOT perturb predictions on peptides outside the new token's
support.

Smoke loop:

```bash
uv sync --extra dev
uv run pytest
```

## Layout

| File | What it owns |
|---|---|
| `api.py` | `RTPredictor`, `FitResult`, the train loop |
| `dataset.py` | `Vocabulary`, tokenizers, `RTDataset`, library loaders |
| `model.py` | `RTModel`, `extend_vocab`, `load_legacy_state_dict` |
| `io.py` | TSV format detection, `apply_psmtag` annotation helper |
| `muon.py` | Muon optimizer for conv weights |
| `train_label_free.py` / `train_psmtag.py` / `predict_library.py` | thin CLI wrappers |
