"""Head-to-head: original timePlex Keras CNN (CNN_LF_iRT.ipynb, faithful) vs
JRIF dilated_cnn (shipping recipe), same data, same train/test split.

Data: JD_LF.tsv label-free library (stand-in for LF_HYE_lib.tsv, which is not
on this machine). Unmodified peptides, iRT-like scale, len <= 30.
"""
import json, sys, time
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATA = "/Users/danielgeiszler/Documents/jmod_tests/20260120_entrapments/libs/JD_LF.tsv"
OUT = "/private/tmp/claude-501/-Users-danielgeiszler-PycharmProjects-JRIF/1282825a-a1ab-41c2-9b86-3c24db373145/scratchpad/bench_keras_vs_jrif.json"
SEED = 123

# ---------------- shared data prep (mirrors the notebook) ----------------
df = (
    pl.scan_csv(DATA, separator="\t")
      .select(["PeptideSequence", "Tr_recalibrated"])
      .collect()
      .group_by("PeptideSequence")
      .agg(pl.col("Tr_recalibrated").mean().alias("rt"))
      .rename({"PeptideSequence": "seq"})
      .sort("seq")  # deterministic order before split
)
seqs = df["seq"].to_list()
rts = df["rt"].to_numpy()
seq_tr, seq_te, y_tr, y_te = train_test_split(seqs, rts, test_size=0.1, random_state=SEED)
print(f"train {len(seq_tr)}  test {len(seq_te)}", flush=True)

def metrics(y, p):
    err = np.abs(np.asarray(p, float) - y)
    return {"r2": float(r2_score(y, p)), "mae": float(err.mean()),
            "d95": float(np.quantile(err, 0.95))}

results = {"n_train": len(seq_tr), "n_test": len(seq_te), "seed": SEED, "data": DATA}

# ---------------- Arm A: original Keras CNN, faithful ----------------
import tensorflow as tf
import random as pyrandom
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(SEED); np.random.seed(SEED); pyrandom.seed(SEED)

AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AAS)}
MAXLEN = 30

def onehot(seq):
    x = np.zeros((MAXLEN, 20), dtype=np.float32)
    for i, aa in enumerate(seq[:MAXLEN]):
        j = AA_IDX.get(aa.upper())
        if j is not None:
            x[i, j] = 1.0
    return x

X_tr = np.stack([onehot(s) for s in seq_tr])
X_te = np.stack([onehot(s) for s in seq_te])

def build_model():
    m = Sequential([
        Input(shape=(MAXLEN, 20)),
        Conv1D(32, 4, activation="swish", padding="same"), MaxPooling1D(1),
        Conv1D(64, 4, activation="swish", padding="same"), MaxPooling1D(1),
        Conv1D(128, 4, activation="swish", padding="same"), MaxPooling1D(2),
        Conv1D(256, 4, activation="swish", padding="same"), MaxPooling1D(2),
        Flatten(),
        Dense(1000, activation="swish"), Dense(256, activation="swish"),
        Dense(128, activation="swish"), Dense(64, activation="swish"),
        Dense(32, activation="swish"), Dense(1, activation="linear"),
    ])
    m.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
    return m

ENSEMBLE = 5
keras_preds, keras_models = [], []
t0 = time.time()
for i in range(ENSEMBLE):
    print(f"--- keras model {i+1}/{ENSEMBLE}", flush=True)
    m = build_model()
    m.fit(X_tr, y_tr, epochs=150, batch_size=32, validation_split=0.1,
          callbacks=[EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
          verbose=2)
    keras_preds.append(m.predict(X_te, verbose=0).flatten())
keras_time = time.time() - t0

per_model = [metrics(y_te, p) for p in keras_preds]
ens = metrics(y_te, np.mean(keras_preds, axis=0))
results["keras_original"] = {
    "params": int(build_model().count_params()),
    "per_model": per_model,
    "ensemble": ens,
    "train_time_s": keras_time,
}
print("keras ensemble:", ens, flush=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

# ---------------- Arm B: JRIF dilated_cnn, shipping recipe ----------------
import pandas as pd
from jrif.api import RTPredictor

train_df = pd.DataFrame({"seq": seq_tr, "rt": y_tr})
test_df = pd.DataFrame({"seq": seq_te})

t0 = time.time()
p = RTPredictor.from_blank(arch="dilated_cnn")
p.fit(train_df, epochs=150, patience=20, lr=1e-3, batch_size=128,
      loss="mixture", selection_metric="mae", seed=SEED, device="cpu")
jrif_time = time.time() - t0
pred = p.predict(test_df)
results["jrif_dilated_cnn"] = {
    "params": int(sum(x.numel() for x in p.model.parameters())),
    "test": metrics(y_te, np.asarray(pred)),
    "train_time_s": jrif_time,
}
print("jrif:", results["jrif_dilated_cnn"]["test"], flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("wrote", OUT)
