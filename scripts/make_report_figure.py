import argparse, json, sys
from pathlib import Path
import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.dataset import read_image
from filament.metrics import IOU_THR, iou_matrix
from filament.postprocess import PostCfg, instances
from filament.rle import decode

def entry_pq(gt, pred):
    m = iou_matrix(gt, pred)
    if m.shape[1] == 0:
        return 0.0
    hit = m > IOU_THR
    den = hit.sum() + 0.5 * (hit.sum(0) == 0).sum() + 0.5 * (hit.sum(1) == 0).sum()
    return float(m[hit].sum() / den) if den else 0.0

def outline(rgb, masks, color):
    for m in masks:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cnts, -1, color, 3)
    return rgb

ap = argparse.ArgumentParser()
ap.add_argument("--probs", required=True); ap.add_argument("--cache", default="cache")
ap.add_argument("--post", required=True); ap.add_argument("--out", default="fig_examples.png")
ap.add_argument("--crop", type=int, default=900)
a = ap.parse_args()
post = PostCfg(**json.load(open(a.post))["post"])
gt = pd.read_csv(Path(a.cache) / "gt.csv")
gt["ann"] = gt["filament_id"].str.split("_", n=1).str[0]
rows = []
for f in sorted(Path(a.probs).glob("*.npy")):
    g = gt[gt["stem"] == f.stem]
    if g.empty: continue
    first = sorted(g["ann"].unique())[0]
    gc = g.loc[g["ann"] == first, "segmentation_rle"].tolist()
    if len(gc) < 3: continue
    pred = [c["counts"] for c in instances(np.load(f).astype(np.float32), post)]
    rows.append((f.stem, entry_pq(gc, pred), gc, pred))
rows.sort(key=lambda r: r[1])
picks = [rows[int(q * (len(rows) - 1))] for q in (0.85, 0.5, 0.15)]
fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5))
for row, (stem, pq, gc, pred), lab in zip(axes, picks, ["good case", "typical case", "hard case"]):
    img = read_image(Path(a.cache) / "img_flat" / f"{stem}.png")
    gm = [decode(c).astype(bool) for c in gc]; pm = [decode(c).astype(bool) for c in pred]
    dens = cv2.blur(np.sum(gm, axis=0).astype(np.float32), (a.crop, a.crop))
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    h, w = img.shape
    y0 = int(np.clip(cy - a.crop // 2, 0, h - a.crop)); x0 = int(np.clip(cx - a.crop // 2, 0, w - a.crop))
    sl = (slice(y0, y0 + a.crop), slice(x0, x0 + a.crop))
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    panels = [base[sl], outline(base.copy(), gm, (0, 200, 0))[sl], outline(base.copy(), pm, (230, 0, 230))[sl]]
    titles = [f"{lab}: input", f"annotation ({len(gm)} filaments)", f"prediction ({len(pm)}), image PQ {pq:.2f}"]
    for ax, p, t in zip(row, panels, titles):
        ax.imshow(p); ax.set_title(t, fontsize=10); ax.axis("off")
plt.tight_layout(); plt.savefig(a.out, dpi=150)
print("saved", a.out, "| PQ:", [round(p[1], 3) for p in picks])
