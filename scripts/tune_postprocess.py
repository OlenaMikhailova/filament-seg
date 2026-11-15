"""Grid-search post-processing on saved validation probabilities, scored with official PQ.

For each (thr, close_k, grow) the candidate instances and their IoU with every GT
annotator-image are computed once; the min_area / min_conf / conf_small filters then
only select columns, so that inner grid is nearly free.

Each validation image uses the average of every run that has a probability map for it:
- runs of different folds  -> out-of-fold tuning on all their images together
    python scripts/tune_postprocess.py --runs runs/r34_f0_e80 runs/r34_f1 runs/r34_f2 runs/r34_f3 runs/r34_f4
- runs of the same fold    -> ensemble tuning (probabilities averaged)
    python scripts/tune_postprocess.py --runs runs/r34_f0 runs/effb4_f0

The outer grid can be narrowed from the command line, e.g. around a known optimum:
    --thrs 0.3 0.35 0.4 --close-ks 5 9 --grows -1 0 1 2
The default --grows 0 and --conf-smalls none reproduce the original 4-parameter search.

Writes best_post.json and postprocess_grid.csv to --out (default: the first run's folder),
and prints PQ per run's images for the best setting (how stable it is across folds).
"""
import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.metrics import IOU_THR, iou_matrix  # noqa: E402
from filament.postprocess import candidate_instances, conf_thresholds  # noqa: E402

THRS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
CLOSE_KS = [0, 5, 9, 15, 25]
MIN_AREAS = [20, 100, 200, 300, 400, 500, 800]  # 20 = the extraction floor
MIN_CONFS = [0.0, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
SMALL_AREA = 1000

_STATE = {}


def _init(runs, gt_path):
    gt = pd.read_csv(gt_path)
    _STATE["runs"] = runs
    _STATE["gt"] = {ai: g["segmentation_rle"].tolist() for ai, g in gt.groupby(gt["filament_id"].str.split("_", n=1).str[0])}


def _load_prob(stem):
    files = [Path(r) / "val_probs" / f"{stem}.npy" for r in _STATE["runs"]]
    return np.mean([np.load(f).astype(np.float32) for f in files if f.exists()], axis=0)


def _per_image(args):
    """For one stem and one (thr, k, grow): candidate stats + IoU matrices vs each annotator."""
    stem, thr, k, g = args
    cands = candidate_instances(_load_prob(stem), thr, k, grow=g)
    area = np.array([c["area"] for c in cands])
    conf = np.array([c["conf"] for c in cands])
    counts = [c["counts"] for c in cands]
    mats = [iou_matrix(gt, counts) for ai, gt in _STATE["gt"].items() if ai.split("-", 1)[1] == stem]
    return stem, thr, k, g, area, conf, mats


def pq_for_filter(entries, min_area, min_conf, conf_small=None):
    tp_sum, tp, fp, fn = 0.0, 0, 0, 0
    for area, conf, mats in entries:
        if len(area):
            keep = (area >= min_area) & (conf >= conf_thresholds(area, min_conf, conf_small, SMALL_AREA))
        else:
            keep = np.zeros(0, bool)
        for m in mats:
            n_gt = m.shape[0]
            mm = m[:, keep] if len(area) else np.zeros((n_gt, 0))
            if mm.shape[1] == 0:
                fn += n_gt
                continue
            hit = mm > IOU_THR
            tp_sum += mm[hit].sum()
            tp += int(hit.sum())
            fp += int((hit.sum(0) == 0).sum())
            fn += int((hit.sum(1) == 0).sum())
    den = tp + 0.5 * fp + 0.5 * fn
    return (tp_sum / den if den else 0.0), tp, fp, fn


def _conf_small(v: str):
    return None if v.lower() == "none" else float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", help="folder for best_post.json (default: first run)")
    ap.add_argument("--thrs", nargs="+", type=float, default=THRS)
    ap.add_argument("--close-ks", nargs="+", type=int, default=CLOSE_KS)
    ap.add_argument("--grows", nargs="+", type=int, default=[0])
    ap.add_argument("--min-areas", nargs="+", type=int, default=MIN_AREAS)
    ap.add_argument("--min-confs", nargs="+", type=float, default=MIN_CONFS)
    ap.add_argument("--conf-smalls", nargs="+", type=_conf_small, default=[None],
                    help="confidence thresholds for instances < 1000 px; 'none' = same as min_conf")
    a = ap.parse_args()

    run_stems = {r: {p.stem for p in (Path(r) / "val_probs").glob("*.npy")} for r in a.runs}
    for r, s in run_stems.items():
        if not s:
            sys.exit(f"no val_probs in {r} (did training finish?)")
    stems = sorted(set().union(*run_stems.values()))
    print(f"{len(stems)} validation images from {len(a.runs)} run(s): "
          + ", ".join(f"{Path(r).name}={len(s)}" for r, s in run_stems.items()))
    outer = list(itertools.product(a.thrs, a.close_ks, a.grows))
    print(f"{len(outer)} (thr, close_k, grow) settings x {len(stems)} images")
    jobs = [(s, t, k, g) for t, k, g in outer for s in stems]
    grouped, by_stem = {}, {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.runs, str(Path(a.cache) / "gt.csv"))) as ex:
        for stem, t, k, g, area, conf, mats in ex.map(_per_image, jobs, chunksize=4):
            grouped.setdefault((t, k, g), []).append((area, conf, mats))
            by_stem[(t, k, g, stem)] = (area, conf, mats)

    results = []
    for (t, k, g), entries in grouped.items():
        for ma, mc, cs in itertools.product(a.min_areas, a.min_confs, a.conf_smalls):
            pq, tp, fp, fn = pq_for_filter(entries, ma, mc, cs)
            results.append({"thr": t, "close_k": k, "grow": g, "min_area": ma, "min_conf": mc,
                            "conf_small": cs, "pq": pq, "tp": tp, "fp": fp, "fn": fn})
    df = pd.DataFrame(results).sort_values("pq", ascending=False)
    out_dir = Path(a.out or a.runs[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "postprocess_grid.csv", index=False)
    print(df.head(15).to_string(index=False))
    best = df.iloc[0]
    cs = None if pd.isna(best.conf_small) else float(best.conf_small)
    post = {"thr": float(best.thr), "close_k": int(best.close_k), "min_area": int(best.min_area),
            "min_conf": float(best.min_conf), "grow": int(best.grow), "conf_small": cs, "small_area": SMALL_AREA}

    checks = [("thr", post["thr"], a.thrs), ("close_k", post["close_k"], a.close_ks), ("grow", post["grow"], a.grows),
              ("min_area", post["min_area"], a.min_areas), ("min_conf", post["min_conf"], a.min_confs)]
    edges = [n for n, v, grid in checks if len(grid) > 1 and v in (min(grid), max(grid))]
    if edges:
        print(f"note: best {', '.join(edges)} on the edge of the grid; consider widening it")
    if len(a.runs) > 1:
        print("PQ with the best setting, per run's validation images:")
        for r, s in run_stems.items():
            ents = [by_stem[(post['thr'], post['close_k'], post['grow'], st)] for st in sorted(s)]
            pq, *_ = pq_for_filter(ents, post["min_area"], post["min_conf"], cs)
            print(f"  {Path(r).name:20s} {len(s):4d} images  PQ {pq:.4f}")
    with open(out_dir / "best_post.json", "w") as f:
        json.dump({"post": post, "val_pq": float(best.pq), "runs": a.runs}, f, indent=2)
    print("best:", post, f"PQ {best.pq:.4f}")


if __name__ == "__main__":
    main()
