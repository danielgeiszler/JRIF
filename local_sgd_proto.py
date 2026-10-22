"""Local SGD prototype: K CPU workers x bs128, parameter averaging every H steps.

Validates quality + wall-clock vs single-process CPU bs128 on the PSMtag set.
Usage: local_sgd_proto.py [single|localsgd] [K] [H]
"""
import sys, json, math, time
sys.path.insert(0, '/Users/danielgeiszler/PycharmProjects/JRIF')
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader

import bench_transfer as B
from jrif.api import RTPredictor, _make_optimizers
from jrif.dataset import RTDataset, Vocabulary, fit_rt_scale
from jrif.loss import LaplaceUniformMixture

MAX_ROUNDS = 200
PATIENCE_ROUNDS = 40
BS = 256


def build_model():
    v = Vocabulary.default()
    v.extend(".[308]", None, fresh=True)
    p = RTPredictor.from_blank(arch="dilated_cnn", unified_termini=True)
    p.model.extend_vocab(new_main_tokens=[(".[308]", None, True)])
    return p


def make_ds(df, med, mad):
    return RTDataset(df, vocab=None or _VOCAB, rt_median=med, rt_mad=mad,
                     terminal_in_seq=True)


def eval_mae(model, loader):
    model.eval()
    errs = []
    with torch.inference_mode():
        for s, nt, ct, y in loader:
            errs.append((model(s, nt, ct) - y).abs())
    return torch.cat(errs).mean().item()


def worker_main(rank, K, H, conn, df_pack, seed):
    torch.set_num_threads(3)
    torch.manual_seed(seed + rank)
    tok, nt, ct, tgt = df_pack
    from torch.utils.data import TensorDataset
    ds = TensorDataset(tok, nt, ct, tgt)
    loader = DataLoader(ds, batch_size=BS, shuffle=True,
                        generator=torch.Generator().manual_seed(seed + rank))
    p = build_model()
    model = p.model
    criterion = LaplaceUniformMixture(init_b=1.0, init_eps=0.05, W=10.0)
    opts, _ = _make_optimizers(model, 1e-3, 1e-4, 10)
    opts[0].add_param_group({"params": list(criterion.parameters())})
    it = iter(loader)
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            break
        _, state, crit_state, lr = msg
        model.load_state_dict(state)
        criterion.load_state_dict(crit_state)
        for o in opts:
            for g in o.param_groups:
                base = g.get("_base_lr")
                if base is None:
                    g["_base_lr"] = g["lr"]; base = g["lr"]
                g["lr"] = base * lr
        model.train()
        for _ in range(H):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader); batch = next(it)
            s, ntb, ctb, y = batch
            loss = criterion(model(s, ntb, ctb), y)
            loss.backward()
            for o in opts:
                o.step(); o.zero_grad()
        conn.send((model.state_dict(), criterion.state_dict()))


def average_states(states):
    out = {}
    for k in states[0]:
        v = states[0][k]
        if v.dtype.is_floating_point:
            out[k] = torch.stack([s[k].float() for s in states]).mean(0).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "localsgd"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    seed = 0
    torch.manual_seed(seed)

    psm = B.load_psmtag()
    tr, va, te = B.split(psm)
    med, mad = fit_rt_scale(tr["rt"].to_numpy().astype("float32"))
    global _VOCAB
    _VOCAB = Vocabulary.default(); _VOCAB.extend(".[308]", None, fresh=True)

    def pack(df):
        ds = RTDataset(df, _VOCAB, rt_median=med, rt_mad=mad, terminal_in_seq=True)
        return (ds.sequences, ds.nterm_mod_ids, ds.cterm_mod_ids, ds.targets)

    tr_pack, va_pack, te_pack = pack(tr), pack(va), pack(te)
    from torch.utils.data import TensorDataset
    va_loader = DataLoader(TensorDataset(*va_pack), batch_size=2048)
    te_loader = DataLoader(TensorDataset(*te_pack), batch_size=2048)

    t0 = time.time()
    if mode == "single":
        torch.set_num_threads(10)
        p = build_model()
        tr_df = tr
        p.fit(tr_df, val_df=va, epochs=150, patience=20, lr=1e-3, batch_size=BS,
              loss="mixture", selection_metric="mae", seed=seed, device="cpu", verbose=False)
        model = p.model
    else:
        ctx = mp.get_context("spawn")
        conns, procs = [], []
        for r in range(K):
            parent, child = ctx.Pipe()
            pr = ctx.Process(target=worker_main, args=(r, K, H, child, tr_pack, seed))
            pr.start(); conns.append(parent); procs.append(pr)
        p = build_model()
        model = p.model
        criterion = LaplaceUniformMixture(init_b=1.0, init_eps=0.05, W=10.0)
        best = (float("inf"), None, -1)
        wait = 0
        for rnd in range(MAX_ROUNDS):
            lr_frac = 0.5 * (1 + math.cos(math.pi * rnd / MAX_ROUNDS))
            lr_frac = min(1.0, (rnd + 1) / 3) * lr_frac  # 3-round warmup
            state, crit_state = model.state_dict(), criterion.state_dict()
            for c in conns:
                c.send(("run", state, crit_state, lr_frac))
            replies = [c.recv() for c in conns]
            model.load_state_dict(average_states([r[0] for r in replies]))
            criterion.load_state_dict(average_states([r[1] for r in replies]))
            vm = eval_mae(model, va_loader)
            if vm < best[0]:
                best = (vm, {k: v.clone() for k, v in model.state_dict().items()}, rnd)
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE_ROUNDS:
                    break
            if rnd % 20 == 0:
                print(f"round {rnd} val_mae(z) {vm:.4f} best {best[0]:.4f}@{best[2]}", flush=True)
        for c in conns: c.send(("stop",))
        for pr in procs: pr.join()
        model.load_state_dict(best[1])

    wall = time.time() - t0
    model.eval()
    errs, preds, ys = [], [], []
    with torch.inference_mode():
        for s, ntb, ctb, y in te_loader:
            pr_ = model(s, ntb, ctb)
            preds.append(pr_); ys.append(y)
    pr_ = torch.cat(preds).numpy() * mad; y = torch.cat(ys).numpy() * mad
    res = B.metrics(y, pr_)
    res.update(wall_s=round(wall, 1), mode=mode, K=K, H=H)
    print(json.dumps(res), flush=True)
    json.dump(res, open(f"/Users/danielgeiszler/PycharmProjects/JRIF/results/localsgd_{mode}_K{K}_H{H}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
