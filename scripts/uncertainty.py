import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.metrics import evaluate
from filament.postprocess import PostCfg, instances
from filament.submission import to_dataframe

def pq_of(a):
    s, tp, fp, fn = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    den = tp + 0.5 * fp + 0.5 * fn
    return (np.where(den > 0, s / np.maximum(den, 1e-9), 0),
            np.where(tp > 0, s / np.maximum(tp, 1e-9), 0),
            np.where(den > 0, tp / np.maximum(den, 1e-9), 0))

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True)
ap.add_argument("--cache", default="cache"); ap.add_argument("--post", required=True)
ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--out", default="uncertainty.csv")
a = ap.parse_args()
gt = pd.read_csv(Path(a.cache) / "gt.csv")
folds = pd.read_csv(Path(a.cache) / "folds.csv").set_index("stem")["fold"]
rng = np.random.default_rng(0); rows = []
for spec in a.run:
    name, rest = spec.split("=", 1)
    dirs, post_path = rest.rsplit(":", 1) if ":" in rest else (rest, a.post)
    post = PostCfg(**json.load(open(post_path))["post"])
    files = [f for d in dirs.split(",") for f in sorted(Path(d).glob("*.npy"))]
    preds = {f.stem: [c["counts"] for c in instances(np.load(f).astype(np.float32), post)] for f in files}
    g = gt[gt["stem"].isin(list(preds))]
    res = evaluate(g[["filament_id", "segmentation_rle"]], to_dataframe(preds), keep_per_entry=True)
    e = res.per_entry
    e["iou_sum"] = e["pq"] * (e["tp"] + 0.5 * e["fp"] + 0.5 * e["fn"])
    per_img = e.groupby("stem")[["iou_sum", "tp", "fp", "fn"]].sum()
    arr = per_img.to_numpy(float)
    pq, sq, rq = pq_of(arr.sum(0))
    fk = folds.reindex(per_img.index).to_numpy()
    fold_pq = [float(pq_of(arr[fk == k].sum(0))[0]) for k in np.unique(fk)]
    bpq = pq_of(arr[rng.integers(0, len(arr), (a.boot, len(arr)))].sum(1))[0]
    lo, hi = np.percentile(bpq, [2.5, 97.5])
    rows.append(dict(run=name, images=len(arr), PQ=pq, SQ=sq, RQ=rq, TP=res.tp, FP=res.fp, FN=res.fn,
                     fold_mean=np.mean(fold_pq), fold_std=np.std(fold_pq), ci_lo=lo, ci_hi=hi))
    print(f"{name:12s} n={len(arr):4d} PQ {pq:.3f} [95% CI {lo:.3f}-{hi:.3f}] SQ {sq:.3f} RQ {rq:.3f} "
          f"TP {res.tp} FP {res.fp} FN {res.fn}"
          + (f" | folds {', '.join(f'{x:.3f}' for x in fold_pq)} -> {np.mean(fold_pq):.3f} +- {np.std(fold_pq):.3f}" if len(fold_pq) > 1 else ""))
pd.DataFrame(rows).to_csv(a.out, index=False); print("saved", a.out)
