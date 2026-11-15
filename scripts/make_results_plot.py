"""Report figure from uncertainty.csv: (a) PQ with 95% bootstrap CI, (b) SQ vs RQ.

  python scripts/make_results_plot.py --csv uncertainty.csv --out fig_results.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {"r34_f0": "U-Net ResNet34", "ridge_f0": "+ ridge/top-hat input", "upp_f0": "UNet++ decoder",
          "lovasz_f0": "+ Lovász fine-tune", "dlv3p_f0": "DeepLabV3+ decoder"}
BASE = "r34_f0"

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="uncertainty.csv")
ap.add_argument("--out", default="fig_results.png")
a = ap.parse_args()

df = pd.read_csv(a.csv)
df = df[df["run"].isin(LABELS)].copy()
df["order"] = df["run"].map(list(LABELS).index)
df = df.sort_values("order").reset_index(drop=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa0a6"})
BLUE, GRAY, ORANGE, MUTED = "#2a78d6", "#6b7280", "#eb6834", "#52514e"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 1.9), dpi=300,
                               gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.08))
ys = list(range(len(df)))[::-1]
base_pq = df.loc[df["run"] == BASE, "PQ"].iloc[0]

# (a) PQ with 95% CI
ax1.axvline(base_pq, color=BLUE, lw=0.8, ls=(0, (3, 2)), zorder=1)
for y, r in zip(ys, df.itertuples()):
    c = BLUE if r.run == BASE else GRAY
    ax1.plot([r.ci_lo, r.ci_hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
    ax1.plot(r.PQ, y, "o", ms=5, color=c, mec="white", mew=1.2, zorder=3)
ax1.set_yticks(ys)
ax1.set_yticklabels([LABELS[r] for r in df["run"]])
ax1.get_yticklabels()[0].set_fontweight("bold")
ax1.set_xlabel("PQ (95% CI)", color=MUTED)
ax1.tick_params(axis="y", length=0)
ax1.spines["left"].set_visible(False)
ax1.grid(axis="x", color="#e6e6e3", lw=0.6, zorder=0)
ax1.set_title("(a) PQ on fold 0", fontsize=8, loc="left")

# (b) SQ vs RQ
for y, r in zip(ys, df.itertuples()):
    ax2.plot([r.RQ, r.SQ], [y, y], color="#d4d4d0", lw=1, zorder=1)
    ax2.plot(r.SQ, y, "s", ms=4.5, color=ORANGE, mec="white", mew=1, zorder=3)
    ax2.plot(r.RQ, y, "o", ms=5, color=BLUE if r.run == BASE else GRAY, mec="white", mew=1.2, zorder=3)
top = df.iloc[0]
ax2.text(top.SQ, ys[0] + 0.42, "SQ", color=MUTED, fontsize=7, ha="center")
ax2.text(top.RQ, ys[0] + 0.42, "RQ", color=MUTED, fontsize=7, ha="center")
ax2.set_yticks(ys)
ax2.set_yticklabels([])
ax2.tick_params(axis="y", length=0)
ax2.spines["left"].set_visible(False)
ax2.grid(axis="x", color="#e6e6e3", lw=0.6, zorder=0)
ax2.set_xlabel("score", color=MUTED)
ax2.set_title("(b) SQ vs RQ", fontsize=8, loc="left")
for ax in (ax1, ax2):
    ax.set_ylim(-0.6, len(df) - 0.3)

fig.savefig(a.out, bbox_inches="tight", facecolor="white")
print("saved", a.out)
