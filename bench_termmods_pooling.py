"""Two architecture experiments on a PSMtag report-lib:

  1. inseq    — N-term mod as a literal main-vocab token at sequence position 0
                (instead of the separate nterm_mod_id channel + head injection).
                NOTE: in this library 100% of peptides carry the N-term tag, so
                the token is constant; this run is a does-it-hurt sanity check,
                not a discrimination test.
  2. attnpool — replace masked GAP with single-query attention pooling over the
                conv outputs.

Baseline: stock RTModel (dilated_cnn), shipping recipe (mixture loss, val-MAE
selection, Muon+AdamW, cosine + warmup). Same peptide-level 80/10/10 split and
robust-z scale for all variants. Metrics reported in the library's native RT
units.

Data: report-library parquet with (tag) / (UniMod:n) notation; converted to
JRIF psmtag format: leading "(tag)" -> ".[308]", "K(tag)" -> "K[308]",
"C(UniMod:4)" -> "C[160]", "M(UniMod:35)" -> "M[147]".
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from jrif.dataset import MAX_LEN, RTDataset, Vocabulary, fit_rt_scale
from jrif.loss import LaplaceUniformMixture
from jrif.model import RTModel
from jrif.muon import Muon

DATA = "/Users/danielgeiszler/Documents/jmod_tests/20260902_apex/report-lib.parquet"

_MOD_MAP = {"(UniMod:4)": "[160]", "(UniMod:35)": "[147]", "(tag)": "[308]"}


def convert_seq(s: str) -> str:
    if s.startswith("(tag)"):
        s = ".[308]" + s[len("(tag)"):]
    for k, v in _MOD_MAP.items():
        s = s.replace(k, v)
    return s


def load_data() -> pl.DataFrame:
    df = pl.from_arrow(pq.read_table(DATA, columns=["Modified.Sequence", "RT"]))
    df = (
        df.group_by("Modified.Sequence").agg(pl.col("RT").mean().alias("rt"))
          .with_columns(pl.col("Modified.Sequence")
                        .map_elements(convert_seq, return_dtype=pl.Utf8).alias("seq"))
          .select(["seq", "rt"])
          .sort("seq")
    )
    return df


def split(df: pl.DataFrame, seed: int = 42):
    seqs = df["seq"].to_list()
    rng = random.Random(seed)
    rng.shuffle(seqs)
    n = len(seqs)
    n_test, n_val = int(n * 0.1), int(n * 0.1)
    test_s, val_s = set(seqs[:n_test]), set(seqs[n_test:n_test + n_val])
    return (
        df.filter(~pl.col("seq").is_in(list(test_s | val_s))),
        df.filter(pl.col("seq").is_in(list(val_s))),
        df.filter(pl.col("seq").is_in(list(test_s))),
    )


# ---------- inseq tokenization: N-term mod as a main token at position 0 ----------

_NTERM = re.compile(r"^\.\[[\d.]+\]")
_MAIN = re.compile(r"[A-Z](?:\[[\d.]+\])?")


def tokenize_inseq(seq: str, vocab: Vocabulary) -> tuple[list[int], int]:
    """Returns (token ids with nterm mod as literal first token, nterm_mod_id=0)."""
    ids = []
    m = _NTERM.match(seq)
    if m:
        ids.append(vocab.tokens[m.group(0)])  # ".[308]" as a main token
        seq = seq[m.end():]
    for t in _MAIN.findall(seq):
        ids.append(vocab.tokens[t])
    return ids, 0


def make_inseq_dataset(df: pl.DataFrame, vocab: Vocabulary, rt_med, rt_mad):
    n = df.height
    tok = np.zeros((n, MAX_LEN), dtype=np.int64)
    for i, s in enumerate(df["seq"].to_list()):
        ids, _ = tokenize_inseq(s, vocab)
        tok[i, :len(ids)] = ids[:MAX_LEN]
    z = ((df["rt"].to_numpy() - rt_med) / rt_mad).astype(np.float32)
    zeros = torch.zeros(n, dtype=torch.long)
    return TensorDataset(torch.from_numpy(tok), zeros, zeros.clone(), torch.from_numpy(z))


# ---------- attention pooling variant ----------

class AttnPoolRTModel(RTModel):
    """RTModel with single-query attention pooling instead of masked GAP."""

    def __init__(self, vocab: Vocabulary, embedding_dim: int = 16):
        super().__init__(vocab, embedding_dim=embedding_dim)
        d = 128
        self.pool_query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.pool_key = nn.Linear(d, d)

    def forward(self, x, nterm_mod_id=None, cterm_mod_id=None):
        B = x.size(0)
        tokens = x
        token_mask = (tokens != 0).float()

        n_term_token = tokens[:, 0]
        lengths = (x != 0).sum(dim=1)
        c_term_idx = (lengths - 1).clamp(min=0)
        c_term_tokens = x.gather(1, c_term_idx.unsqueeze(1)).squeeze(1)
        n_term_emb = self.n_term_embedding(n_term_token)
        c_term_emb = self.c_term_embedding(c_term_tokens)

        if nterm_mod_id is None:
            nterm_mod_id = torch.zeros(B, dtype=torch.long, device=tokens.device)
        if cterm_mod_id is None:
            cterm_mod_id = torch.zeros(B, dtype=torch.long, device=tokens.device)
        n_mod = self.nterm_mod_embedding(nterm_mod_id)
        c_mod = self.cterm_mod_embedding(cterm_mod_id)

        phys = self.physico_features[x]
        emb = self.embedding(x)
        z = torch.cat([phys, emb], dim=2).permute(0, 2, 1)
        z = self.convs(z)                                   # (B, 128, L)

        # Attention pooling: scores from projected keys vs a learned query.
        keys = self.pool_key(z.permute(0, 2, 1))            # (B, L, 128)
        scores = keys @ self.pool_query / math.sqrt(keys.size(-1))  # (B, L)
        scores = scores.masked_fill(tokens == 0, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(1)       # (B, 1, L)
        z = (z * w).sum(dim=2)                              # (B, 128)

        seq_lengths = token_mask.sum(dim=1, keepdim=True).clamp(min=1)
        gravy = (self.hydro_table[tokens] * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        net_charge = (self.charge_table[tokens] * token_mask).sum(dim=1, keepdim=True)
        frac_aromatic = (self.aromatic_table[tokens] * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        frac_hydrophobic = (self.hydrophobic_table[tokens] * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        norm_seq_len = seq_lengths / float(MAX_LEN)

        feats = torch.cat([
            z, n_term_emb, c_term_emb, n_mod, c_mod,
            gravy, net_charge, frac_aromatic,
            norm_seq_len, frac_hydrophobic,
        ], dim=1)
        return self.head(feats).squeeze(1)


# ---------- shared training loop (shipping recipe) ----------

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def evaluate(model, loader, device, rt_mad):
    model.eval()
    preds, tgts = [], []
    with torch.inference_mode():
        for s, nt, ct, y in loader:
            p = model(s.to(device), nt.to(device), ct.to(device)).cpu()
            preds.append(p); tgts.append(y)
    p = torch.cat(preds); y = torch.cat(tgts)
    err = (p - y).abs()
    ss_res = ((y - p) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    return {
        "mae": err.mean().item() * rt_mad,          # native units
        "d95": err.quantile(0.95).item() * rt_mad,
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 0.0,
    }


def train(model, train_ds, val_ds, test_ds, rt_mad, *, seed, device,
          epochs=150, patience=20, lr=1e-3, batch_size=1024):
    set_seed(seed)
    model = model.to(device)
    criterion = LaplaceUniformMixture(init_b=1.0, init_eps=0.05, W=10.0).to(device)

    muon_params, adamw_params = [], []
    for name, pr in model.named_parameters():
        if "convs." in name and "conv.weight" in name and pr.ndim >= 2:
            muon_params.append(pr)
        else:
            adamw_params.append(pr)
    adamw_params += list(criterion.parameters())
    opts = [torch.optim.AdamW(adamw_params, lr=lr, weight_decay=1e-4),
            Muon(muon_params, lr=lr * 5, momentum=0.95)]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=epochs) for o in opts]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

    best_mae, best_state, best_epoch, wait = float("inf"), None, -1, 0
    warmup = 3
    for epoch in range(epochs):
        model.train()
        wf = min(1.0, (epoch + 1) / warmup)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g.get("initial_lr", g["lr"]) * wf if epoch < warmup else g["lr"]
        for s, nt, ct, y in train_loader:
            loss = criterion(model(s.to(device), nt.to(device), ct.to(device)), y.to(device))
            loss.backward()
            for o in opts:
                o.step(); o.zero_grad()
        if epoch >= warmup:
            for sc in scheds:
                sc.step()
        vm = evaluate(model, val_loader, device, rt_mad)
        if vm["mae"] < best_mae:
            best_mae, best_epoch, wait = vm["mae"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
        print(f"    ep {epoch+1:3d} val_mae {vm['mae']:.4f} (best {best_mae:.4f} @ {best_epoch+1})", flush=True)
    model.load_state_dict(best_state)
    out = evaluate(model, test_loader, device, rt_mad)
    out.update(best_val_mae=best_mae, best_epoch=best_epoch + 1,
               params=sum(p.numel() for p in model.parameters()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results/bench_termmods_pooling.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    df = load_data()
    train_df, val_df, test_df = split(df)
    print(f"train {train_df.height} val {val_df.height} test {test_df.height}", flush=True)
    rt_med, rt_mad = fit_rt_scale(train_df["rt"].to_numpy().astype("float32"))
    print(f"rt scale: median {rt_med:.4f} mad {rt_mad:.4f}", flush=True)

    results = {"n_train": train_df.height, "n_val": val_df.height,
               "n_test": test_df.height, "rt_median": rt_med, "rt_mad": rt_mad,
               "data": DATA, "variants": {}}

    def std_ds(d):
        return RTDataset(d, Vocabulary.default(), rt_median=rt_med, rt_mad=rt_mad)

    for seed in args.seeds:
        print(f"=== seed {seed}", flush=True)

        print("  baseline", flush=True)
        r = train(RTModel(Vocabulary.default()), std_ds(train_df), std_ds(val_df),
                  std_ds(test_df), rt_mad, seed=seed, device=device)
        results["variants"].setdefault("baseline", []).append(r)

        print("  inseq", flush=True)
        v = Vocabulary.default()
        v.extend(".[308]", None, fresh=True)
        r = train(RTModel(v),
                  make_inseq_dataset(train_df, v, rt_med, rt_mad),
                  make_inseq_dataset(val_df, v, rt_med, rt_mad),
                  make_inseq_dataset(test_df, v, rt_med, rt_mad),
                  rt_mad, seed=seed, device=device)
        results["variants"].setdefault("inseq", []).append(r)

        print("  attnpool", flush=True)
        r = train(AttnPoolRTModel(Vocabulary.default()), std_ds(train_df), std_ds(val_df),
                  std_ds(test_df), rt_mad, seed=seed, device=device)
        results["variants"].setdefault("attnpool", []).append(r)

        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"  (saved through seed {seed})", flush=True)

    for name, runs in results["variants"].items():
        maes = [r["mae"] for r in runs]
        print(f"{name:9s} test MAE {np.mean(maes):.4f} ± {np.std(maes):.4f}  "
              f"(n={len(maes)}, params {runs[0]['params']})")


if __name__ == "__main__":
    main()


# ---------- inseq_clean: sequence is the ONLY terminus representation ----------

class InSeqCleanRTModel(RTModel):
    """RTModel with all terminal head injections removed: no n/c-term residue
    embeddings, no n/c-term mod embeddings. Terminal mods enter only as
    main-vocab sequence tokens. Head input: 128 + 5 globals = 133."""

    def __init__(self, vocab: Vocabulary, embedding_dim: int = 16):
        super().__init__(vocab, embedding_dim=embedding_dim)
        head_in = 128 + 3 + 1 + 1
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x, nterm_mod_id=None, cterm_mod_id=None):
        tokens = x
        mask = (x != 0).unsqueeze(1)
        token_mask = (tokens != 0).float()

        phys = self.physico_features[x]
        emb = self.embedding(x)
        z = torch.cat([phys, emb], dim=2).permute(0, 2, 1)
        z = self.convs(z)
        z = (z * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)

        seq_lengths = token_mask.sum(dim=1, keepdim=True).clamp(min=1)
        gravy = (self.hydro_table[tokens] * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        net_charge = (self.charge_table[tokens] * token_mask).sum(dim=1, keepdim=True)
        frac_aromatic = (self.aromatic_table[tokens] * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        frac_hydrophobic = (self.hydrophobic_table[tokens] * token_mask).sum(dim=1, keepdim=True) / seq_lengths
        norm_seq_len = seq_lengths / float(MAX_LEN)

        feats = torch.cat([
            z, gravy, net_charge, frac_aromatic, norm_seq_len, frac_hydrophobic,
        ], dim=1)
        return self.head(feats).squeeze(1)


def main_clean():
    import sys
    device = torch.device("mps")
    df = load_data()
    train_df, val_df, test_df = split(df)
    rt_med, rt_mad = fit_rt_scale(train_df["rt"].to_numpy().astype("float32"))
    out_path = Path("results/bench_termmods_pooling.json")
    results = json.loads(out_path.read_text())
    for seed in (0, 1, 2):
        print(f"=== inseq_clean seed {seed}", flush=True)
        v = Vocabulary.default()
        v.extend(".[308]", None, fresh=True)
        r = train(InSeqCleanRTModel(v),
                  make_inseq_dataset(train_df, v, rt_med, rt_mad),
                  make_inseq_dataset(val_df, v, rt_med, rt_mad),
                  make_inseq_dataset(test_df, v, rt_med, rt_mad),
                  rt_mad, seed=seed, device=device)
        results["variants"].setdefault("inseq_clean", []).append(r)
        out_path.write_text(json.dumps(results, indent=2))
    for name, runs in results["variants"].items():
        maes = [r["mae"] for r in runs]
        print(f"{name:12s} test MAE {np.mean(maes):.5f} ± {np.std(maes):.5f} (n={len(maes)})")
