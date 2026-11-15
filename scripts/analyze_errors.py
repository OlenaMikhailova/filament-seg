"""Where does PQ get lost? Error breakdown on out-of-fold validation probabilities (CPU only).

  python scripts/analyze_errors.py --runs runs/r34_f0_e80 runs/r34_f1 runs/r34_f2 runs/r34_f3 runs/r34_f4 \
      --post runs/oof_r34/best_post.json --cache cache

For every GT filament (per annotator) it reports why it was missed:
  - fragmented : touched by 2+ predictions, none with IoU > 0.5 (split into pieces)
  - merged     : its best prediction also covers other GT filaments (glued to a neighbour)
  - poor_shape : one prediction overlaps it but IoU <= 0.5 (too thin / too fat / partial)
  - not_found  : no prediction touches it at all (missed or filtered out)
and why predictions are false positives (no GT at all vs. overlapping but bad IoU),
broken down by filament size. The biggest bucket says what to fix first.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.metrics import IOU_THR, iou_matrix  # noqa: E402
from filament.postprocess import PostCfg, instances  # noqa: E402
from filament.rle import to_coco  # noqa: E402

SIZE_BINS = [0, 500, 1000, 2500, 6000, 10**9]
SIZE_LABELS = ["<500", "500-1k", "1k-2.5k", "2.5k-6k", ">6k"]


def size_bin(a):
    return SIZE_LABELS[int(np.searchsorted(SIZE_BINS, a, side="right")) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()

    post = PostCfg(**json.load(open(a.post))["post"])
    gt = pd.read_csv(Path(a.cache) / "gt.csv")
    gt["ann_img"] = gt["filament_id"].str.split("_", n=1).str[0]
    run_files = {}
    for r in a.runs:
        for f in (Path(r) / "val_probs").glob("*.npy"):
            run_files.setdefault(f.stem, []).append(f)
    print(f"{len(run_files)} images, post {post.to_dict()}")

    gt_rows, fp_rows = [], []
    for i, (stem, files) in enumerate(sorted(run_files.items())):
        prob = np.mean([np.load(f).astype(np.float32) for f in files], axis=0)
        pred = [c["counts"] for c in instances(prob, post)]
        pred_area = [int(mask_util.area(to_coco(c))) for c in pred]
        pred_tp = np.zeros(len(pred), bool)
        pred_any = np.zeros(len(pred), bool)
        for ai, g in gt[gt["stem"] == stem].groupby("ann_img"):
            gc = g["segmentation_rle"].tolist()
            areas = [int(mask_util.area(to_coco(c))) for c in gc]
            m = iou_matrix(gc, pred)  # (n_gt, n_pred)
            if m.shape[1]:
                pred_tp |= (m > IOU_THR).any(0)
                pred_any |= (m > 0).any(0)
            for k in range(len(gc)):
                row = m[k] if m.shape[1] else np.zeros(0)
                if (row > IOU_THR).any():
                    cause = "matched"
                elif (row > 0).sum() >= 2:
                    cause = "fragmented"
                elif (row > 0).sum() == 1:
                    j = int(np.argmax(row))
                    cause = "merged" if (m[:, j] > 0).sum() >= 2 else "poor_shape"
                else:
                    cause = "not_found"
                gt_rows.append({"size": size_bin(areas[k]), "cause": cause,
                                "iou": float(row.max()) if len(row) else 0.0})
        for j in range(len(pred)):
            if not pred_tp[j]:
                fp_rows.append({"size": size_bin(pred_area[j]),
                                "cause": "bad_iou_overlap" if pred_any[j] else "no_gt_at_all"})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} images")

    g = pd.DataFrame(gt_rows)
    f = pd.DataFrame(fp_rows)
    order = ["matched", "poor_shape", "fragmented", "merged", "not_found"]
    print("\nGT filaments (one row per annotator): share by cause")
    print((g["cause"].value_counts(normalize=True).reindex(order).fillna(0) * 100).round(1).to_string())
    print("\nGT cause by filament size (% of row)")
    t = pd.crosstab(g["size"], g["cause"], normalize="index").reindex(index=SIZE_LABELS, columns=order).fillna(0)
    t["n"] = g["size"].value_counts().reindex(SIZE_LABELS).fillna(0).astype(int)
    print((t.drop(columns="n") * 100).round(1).assign(n=t["n"]).to_string())
    pm = g.loc[g["cause"] == "poor_shape", "iou"]
    if len(pm):
        print(f"\npoor_shape IoU: median {pm.median():.2f}, share in 0.4-0.5 (near misses) {(pm > 0.4).mean() * 100:.0f}%")
    print("\nFalse positives: share by cause and size")
    if len(f):
        print(pd.crosstab(f["size"], f["cause"]).reindex(SIZE_LABELS).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
