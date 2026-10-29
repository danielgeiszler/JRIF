# JRIF — agent guardrails

## Architecture in one paragraph

`RTPredictor` (`src/jrif/api.py`) is the public DataFrame API. It owns one
`RTModel` (a dilated CNN) plus a `Vocabulary` (token-id maps + lineage) plus
a robust z-score scale `(rt_median, rt_mad)`. The model's embedding tables
and physico/global lookup buffers are sized from the Vocabulary at build
time; `RTModel.extend_vocab(...)` grows them in-place when fine-tune data
contains modification tokens the base model has not seen. Modifications are
always detected from the data, never declared: `fit`/`calibrate` log every
mod they see and whether it is new. A new residue mod's rows inherit from its
residue; a new terminal mod has no parent and starts at zero.

## Hard rules

- **Never delete files without explicit permission.** This includes `.legacy`
  backups, old checkpoints, generated artifacts, anything. "Auto mode" /
  "minimize interruptions" does NOT cover destructive operations. If a file
  looks redundant or stale, leave it alone and ask. Same goes for `rm -rf`,
  overwriting non-version-controlled artifacts, dropping checkpoints during
  refactors, or any other action where the file does not come back from
  `git checkout`.


- **The vocabulary lives in the checkpoint.** Never hardcode a vocab size
  outside `Vocabulary.default()`. Anything that needs
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

- **Shipped pretrained checkpoints output z-scores, not iRT.** The bundled
  checkpoints (`src/jrif/checkpoints/*.pt`) ship with `rt_median=0` and `rt_mad=1`, so their back-transform
  is the identity. The original training datasets had iRT-like scales, but
  those scales are arbitrary to downstream callers — z-score output is the
  canonical contract for these checkpoints. Don't restore the arbitrary
  iRT scale onto a shipped checkpoint. After a user calls `.fit(my_df)` on
  their own data, the predictor adopts `my_df`'s scale and `predict`
  returns numbers back in those units.

- **One representation: termini are tokens.** Every peptide is tokenized
  with a terminus token at both ends — its N-terminal mod (`.[308]`) or
  `<nterm>` at position 0, and `<cterm>` after the last residue (a future
  C-terminal mod token takes its place). There are no separate terminal
  channels or head features (head is 133-dim). Don't add alternative
  representations behind flags; when the representation changes, retrain
  the shipped checkpoints instead of keeping a compatibility path.

- **Checkpoints carry `format_version` (currently 3).** `from_checkpoint`
  accepts only the current version and tells the user to retrain otherwise.
  Bump `CHECKPOINT_FORMAT_VERSION` whenever the input representation or
  architecture changes.

- **One modification notation, as written.** Tokens are
  `<site>[<signed delta>]` (site = residue, `n` or `c`) keeping the precision
  the user wrote; known chemistry (UniMod table, LEGACY_ALIASES) is named at
  4 decimals. Written mods match vocab tokens at the coarser precision; the
  Vocabulary refuses two tokens on a site that agree that way, and an
  ambiguous match raises. Users always get back what they gave: logs and
  errors quote their spelling, never only our canonical name. Never invent
  precision, and never add a rename-on-load path for vocab names — rename
  the shipped artifacts instead.

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
| `api.py` | `RTPredictor` (`fit`/`calibrate`/`predict`/`save`), `SHIPPING_RECIPE`, train loop |
| `im.py` | `IMModel` / `IMDataset` / `IMPredictor` (charge as input channel) |
| `dataset.py` | `Vocabulary`, tokenizers, `RTDataset`, library loaders |
| `model.py` | `RTModel`, `extend_vocab`, fast conv paths |
| `loss.py` | `LaplaceUniformMixture` (shipping loss) + alternatives |
| `io.py` | TSV format detection, `apply_psmtag` annotation helper |
| `muon.py` | Muon optimizer for conv weights |
| `training/train_label_free.py` / `train_psmtag.py` / `train_im.py`, `predict_library.py` | thin CLI wrappers |

Benchmark scripts and experiment results were pruned for production; they
live in git history at commit `ee20581` and earlier. The architecture
registry is dilated_cnn-only; the xformer variants were removed with them.
