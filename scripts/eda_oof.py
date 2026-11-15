"""EDA of out-of-fold performance: where is PQ lost, and how close are we to the annotators?

  python scripts/eda_oof.py --runs runs/r34_f0_e80 runs/r34_f1 runs/r34_f2 runs/r34_f3 runs/r34_f4 \
      --post runs/oof_r34/best_post.json --cache cache --out runs/eda_oof

Prints pooled PQ of the model broken down by
  - GONG observatory (last two letters of the file name: Bh, Ch, Lh, Mh, Th, Uh)
  - year (solar cycle phase)
  - number of annotators of the image
  - image sharpness (variance of the Laplacian inside the disk, in quartiles)
  - number of annotated filaments in the image (quartiles)
and the inter-annotator agreement: on images with 2-3 annotators, each annotator's masks
are scored as a "prediction" against every other annotator (official PQ rules), next to
the model's PQ on exactly the same annotator-images. That is a practical ceiling estimate.
Writes per-image results to <out>/per_image.csv.
"""
import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.metrics import IOU_THR, iou_matrix  # noqa: E402
from filament.postprocess import PostCfg, instances  # noqa: E402

_S = {}


def _init(runs, cache, post, preprocess):
    gt = pd.read_csv(Path(cache) / "gt.csv")
    gt["ann_img"] = gt["filament_id"].str.split("_", n=1).str[0]
    _S["gt"] = {s: {ai: g2["segmentation_rle"].tolist() for ai, g2 in g.groupby("ann_img")}
                for s, g in gt.groupby("stem")}
    _S["runs"], _S["post"] = runs, PostCfg(**post)
    _S["img_dir"] = Path(cache) / f"img_{preprocess}"


def counts(m):
    """(tp_iou_sum, tp, fp, fn) for one GT-vs-pred IoU matrix."""
    n_gt, n_pr = m.shape
    if n_pr == 0:
        return 0.0, 0, 0, n_gt
    if n_gt == 0:
        return 0.0, 0, n_pr, 0
    hit = m > IOU_THR
    return float(m[hit].sum()), int(hit.sum()), int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum())


def _job(stem):
    files = [Path(r) / "val_probs" / f"{stem}.npy" for r in _S["runs"]]
    prob = np.mean([np.load(f).astype(np.float32) for f in files if f.exists()], axis=0)
    pred = [c["counts"] for c in instances(prob, _S["post"])]
    img = cv2.imread(str(_S["img_dir"] / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    disk = img > 0
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)
    sharp = float(lap[cv2.erode(disk.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0].var())

    anns = _S["gt"].get(stem, {})
    model_rows = []
    for ai, g in anns.items():
        s, tp, fp, fn = counts(iou_matrix(g, pred))
        model_rows.append({"stem": stem, "ann_img": ai, "n_gt": len(g), "n_pred": len(pred),
                           "s": s, "tp": tp, "fp": fp, "fn": fn, "sharp": sharp})
    human_rows = []
    for a, b in itertools.permutations(anns, 2):  # a = "prediction", b = ground truth
        s, tp, fp, fn = counts(iou_matrix(anns[b], anns[a]))
        human_rows.append({"stem": stem, "gt_ann": b, "s": s, "tp": tp, "fp": fp, "fn": fn})
    return model_rows, human_rows


def pq(df):
    den = df.tp.sum() + 0.5 * df.fp.sum() + 0.5 * df.fn.sum()
    return df.s.sum() / den if den else 0.0


def table(df, col, title):
    rows = []
    for k, g in df.groupby(col, observed=True):
        rows.append({col: k, "entries": len(g), "images": g.stem.nunique(), "PQ": round(pq(g), 4),
                     "TP": int(g.tp.sum()), "FP": int(g.fp.sum()), "FN": int(g.fn.sum())})
    print(f"\n{title}\n" + pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--preprocess", default="flat")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="runs/eda_oof")
    a = ap.parse_args()
    post = json.load(open(a.post))["post"]
    stems = sorted(set().union(*[{p.stem for p in (Path(r) / "val_probs").glob("*.npy")} for r in a.runs]))
    print(f"{len(stems)} images, post {post}")

    model_rows, human_rows = [], []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.runs, a.cache, post, a.preprocess)) as ex:
        for i, (m, h) in enumerate(ex.map(_job, stems, chunksize=4)):
            model_rows += m
            human_rows += h
            if (i + 1) % 100 == 0:
                print(f"  {i + 1} images")
    df, hu = pd.DataFrame(model_rows), pd.DataFrame(human_rows)
    df["station"] = df.stem.str[-2:]
    df["year"] = df.stem.str[:4]
    df["n_annotators"] = df.groupby("stem").ann_img.transform("nunique")
    img_level = df.drop_duplicates("stem").set_index("stem")
    df["sharpness_q"] = df.stem.map(pd.qcut(img_level.sharp, 4, labels=["Q1 blurry", "Q2", "Q3", "Q4 sharp"]))
    df["n_filaments_q"] = pd.qcut(df.n_gt, 4, duplicates="drop").astype(str)

    print(f"\nOverall model PQ (pooled, {len(df)} annotator-images): {pq(df):.4f}")
    table(df, "station", "By GONG observatory")
    table(df, "year", "By year")
    table(df, "n_annotators", "By number of annotators of the image")
    table(df, "sharpness_q", "By image sharpness (Laplacian variance quartile)")
    table(df, "n_filaments_q", "By number of annotated filaments in the image")

    if len(hu):
        multi = df[df.ann_img.isin(hu.gt_ann)]
        print("\nInter-annotator agreement (images with 2-3 annotators, official PQ rules)")
        print(f"  annotator vs annotator : PQ {pq(hu):.4f}  ({len(hu)} pairs on {hu.stem.nunique()} images)")
        print(f"  model vs annotator     : PQ {pq(multi):.4f}  (same {len(multi)} annotator-images)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_image.csv", index=False)
    hu.to_csv(out / "inter_annotator.csv", index=False)
    print(f"\nsaved {out / 'per_image.csv'} and {out / 'inter_annotator.csv'}")


if __name__ == "__main__":
    main()