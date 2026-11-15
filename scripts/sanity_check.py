"""Checks to run once after prepare.py, before any training:

1. RLE round trip: GT polygons -> counts -> mask -> counts is identical
2. Our fast PQ equals the dense notebook-style PQ on real GT (with perturbed predictions)
3. Submitting one annotator's GT as the prediction scores PQ 1.0 on images with a single annotator

  python scripts/sanity_check.py --cache cache
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.metrics import evaluate, evaluate_dense  # noqa: E402
from filament.rle import decode, encode  # noqa: E402
from filament.submission import to_dataframe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--n", type=int, default=12, help="images to use")
    a = ap.parse_args()
    gt = pd.read_csv(Path(a.cache) / "gt.csv")
    ann = gt["filament_id"].str.split("_", n=1).str[0]
    rng = np.random.default_rng(0)

    # 1. round trip
    for c in gt["segmentation_rle"].sample(50, random_state=0):
        assert encode(decode(c)) == c
    print("[ok] RLE round trip")

    # 2. fast vs dense on perturbed predictions (eroded / dilated / dropped / shifted)
    stems = gt["stem"].drop_duplicates().sample(min(a.n, gt["stem"].nunique()), random_state=0).tolist()
    sub_gt = gt[gt["stem"].isin(stems)][["filament_id", "segmentation_rle"]]
    preds = {}
    for s in stems:
        first = sorted(ann[gt["stem"] == s].unique())[0]
        out = []
        for c in gt.loc[(gt["stem"] == s) & (ann == first), "segmentation_rle"]:
            if rng.random() < 0.15:
                continue
            m = decode(c)
            op = rng.integers(3)
            k = np.ones((int(rng.integers(3, 15)),) * 2, np.uint8)
            m = cv2.erode(m, k) if op == 0 else cv2.dilate(m, k) if op == 1 else np.roll(m, int(rng.integers(-8, 8)), 1)
            if m.any():
                out.append(m)
        # make disjoint as in a real submission
        taken = np.zeros((2048, 2048), bool)
        final = []
        for m in out:
            m = m.astype(bool) & ~taken
            taken |= m
            if m.any():
                final.append(encode(m))
        preds[s] = final
    pred_df = to_dataframe(preds)
    fast, dense = evaluate(sub_gt, pred_df).pq, evaluate_dense(sub_gt, pred_df)
    assert abs(fast - dense) < 1e-9, (fast, dense)
    print(f"[ok] fast PQ == dense PQ ({fast:.6f})")

    # 3. perfect prediction on single-annotator images
    per_stem = gt.groupby("stem")["annotator_image"].nunique()
    singles = per_stem[per_stem == 1].index[: a.n].tolist()
    if singles:
        g = gt[gt["stem"].isin(singles)]
        perfect = to_dataframe({s: g.loc[g["stem"] == s, "segmentation_rle"].tolist() for s in singles})
        r = evaluate(g[["filament_id", "segmentation_rle"]], perfect)
        print(f"[{'ok' if r.pq > 0.999 else 'CHECK'}] GT as prediction on single-annotator images: {r.summary()}")
        print("      (below 1.0 only if one annotator drew overlapping filaments)")


if __name__ == "__main__":
    main()
