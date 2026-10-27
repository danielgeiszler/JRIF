"""Regenerate the summary figures in results/ from the benchmark JSONs.

Usage: uv run --no-project --with matplotlib python make_figures.py
Outputs: results/fig1_vs_timeplex.png, results/mae_vs_data.png,
         results/fig2_calibrate.png, results/fig3_im_charge.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).parent / "results"
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#1a1a19", "#6f6e66"


def style(ax):
    ax.grid(axis="y", color="#e5e4dc", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)


# ---- fig 1: RT accuracy vs timePlex baseline ----
k = json.load(open(R / "bench_keras_vs_jrif.json"))
fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=160)
names = ["timePlex baseline\nbest single\n(2.3M params)",
         "timePlex baseline\n5-model ensemble\n(11M params)",
         "JRIF dilated CNN\nsingle model\n(162k params)"]
vals = [min(m["mae"] for m in k["keras_original"]["per_model"]),
        k["keras_original"]["ensemble"]["mae"],
        k["jrif_dilated_cnn"]["test"]["mae"]]
bars = ax.bar(names, vals, color=[MUTED, ORANGE, BLUE], width=0.62)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom", fontsize=10, color=INK)
ax.set_ylabel("Test MAE (iRT units) — lower is better", color=INK)
ax.set_title("RT prediction vs timePlex baseline (identical data & split)",
             loc="left", fontsize=11, color=INK)
style(ax); ax.set_ylim(0, 2.2)
fig.tight_layout(); fig.savefig(R / "fig1_vs_timeplex.png", facecolor="white")

# ---- fig: MAE vs fine-tuning data (no saturation) ----
ft = json.load(open(R / "q10_titration_s0.json"))
sc = json.load(open(R / "q11_scratch_titration_s0.json"))
Ns = [500, 2000, 5000, 20000, 78190]
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
ax.plot(Ns, [ft[f"n{n}"]["mae"] for n in Ns], color=BLUE, lw=2, marker="o", ms=6,
        label="Fine-tuned from PSMtag base")
ax.plot(Ns, [sc[f"n{n}"]["mae"] for n in Ns], color=ORANGE, lw=2, marker="o", ms=6,
        label="Trained from scratch")
ax.set_xscale("log"); ax.set_xticks(Ns)
ax.set_xticklabels(["500", "2k", "5k", "20k", "78k"]); ax.minorticks_off()
ax.set_xlabel("Fine-tuning peptides (log scale)", color=INK)
ax.set_ylabel("Test MAE (native RT units)", color=INK)
ax.set_title("More IDs keep helping (no plateau); pretrained base pays off\n"
             "exactly where typical experiments live", loc="left", fontsize=11, color=INK)
ax.axvspan(500, 5000, color=BLUE, alpha=0.06)
ax.annotate("typical experiment\n(500–5k IDs):\nbase worth 1.4–2x", (1500, 2.9),
            ha="center", fontsize=8.5, color=INK)
ax.annotate("data-rich: curves merge,\nbut neither has flattened", (45000, 1.35),
            ha="center", fontsize=8.5, color=MUTED)
ax.legend(frameon=False, loc="upper right", fontsize=9)
style(ax); ax.set_ylim(0, 4.1)
fig.tight_layout(); fig.savefig(R / "mae_vs_data.png", facecolor="white")

# ---- fig 2: calibrate() regimes ----
hyb = json.load(open(R / "q14_hybrid_s0.json"))
SPAN = 88.82  # reinj test-set 95% RT span
Ns2 = [100, 500, 2000]
iso_v = [2.07, 1.95, 1.96]  # measured q13 run (its JSON was truncated by a crash)
ft_v = [100 * hyb[f"ft_n{n}"]["mae"] / SPAN for n in Ns2]
fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=160)
ax.plot(Ns2, iso_v, color=AQUA, lw=2, marker="o", ms=6,
        label="Frozen model + isotonic warp (no training)")
ax.plot(Ns2, ft_v, color=BLUE, lw=2, marker="o", ms=6, label="Full fine-tune")
ax.set_xscale("log"); ax.set_xticks(Ns2)
ax.set_xticklabels(["100", "500", "2k"]); ax.minorticks_off()
ax.set_xlabel("Identified peptides available for calibration", color=INK)
ax.set_ylabel("Test MAE (% of gradient span)", color=INK)
ax.set_title("calibrate(): auto-selects the right regime per data size",
             loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8.5)
style(ax); ax.set_ylim(0, 3.6)
fig.tight_layout(); fig.savefig(R / "fig2_calibrate.png", facecolor="white")

# ---- fig 3: IM charge injection ----
q = json.load(open(R / "q21_im_charge_s0.json"))
fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=160)
modes = [("none", "No charge", MUTED), ("head", "Charge at head", ORANGE),
         ("input", "Charge as input channel", BLUE)]
zs = [2, 3, 4]; w = 0.26
for i, (m, lab, c) in enumerate(modes):
    ax.bar([x + (i - 1) * w for x in range(3)],
           [1000 * q[m][f"mae_z{z}"] for z in zs], width=w, color=c, label=lab)
ax.set_xticks(range(3)); ax.set_xticklabels([f"z={z}" for z in zs])
ax.set_ylabel("IM test MAE (1/K0 × 10⁻³)", color=INK)
ax.set_title("Ion-mobility model: where to inject charge", loc="left",
             fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8.5)
style(ax)
fig.tight_layout(); fig.savefig(R / "fig3_im_charge.png", facecolor="white")
print("wrote 4 figures to", R)
