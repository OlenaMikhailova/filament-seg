"""Train and evaluate the second-stage instance rescorer on out-of-fold predictions (CPU).

  python scripts/train_rescorer.py --runs runs/r34_f0_e80 runs/r34_f1 runs/r34_f2 runs/r34_f3 runs/r34_f4 \
      --cache cache --settings 0.3:5 0.4:9 --out runs/rescore_r34

For each (thr:close_k) setting:
1. extract candidates and features on every OOF image (probabilities averaged over the
   runs that have that image, so an ensemble of folds/architectures works too);
2. label each candidate with the fraction of annotators for whom it is a true positive;
3. train a gradient-boosting regressor with 5-fold cross-validation grouped by the
   training fold, so every candidate is scored by a rescorer that never saw its image;
4. tune the score / min-area thresholds for PQ on those out-of-fold scores and compare with
   the plain mean-probability filter on exactly the same candidates.
The best setting is refit on all images and saved as <out>/rescorer.joblib, plus
<out>/best_post.json that scripts/predict.py understands.
"""
import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.metrics import IOU_THR, iou_matrix  # noqa: E402
from filament.rescore import FEATURES, candidate_features  # noqa: E402

SCORE_THRS = np.round(np.arange(0.05, 0.91, 0.025), 3)
MIN_AREAS = [20, 100, 200, 300]
BASE_CONFS = [0.0, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
_S = {}


def _init(runs, cache, preprocess):
    gt = pd.read_csv(Path(cache) / "gt.csv")
    _S["runs"], _S["img_dir"] = runs, Path(cache) / f"img_{preprocess}"
    _S["gt"] = {ai: g["segmentation_rle"].tolist() for ai, g in gt.groupby(gt["filament_id"].str.split("_", n=1).str[0])}


def _job(args):
    stem, thr, k = args
    files = [Path(r) / "val_probs" / f"{stem}.npy" for r in _S["runs"]]
    prob = np.mean([np.load(f).astype(np.float32) for f in files if f.exists()], axis=0)
    img = cv2.imread(str(_S["img_dir"] / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    counts, X = candidate_features(prob, img, thr, k)
    mats = [iou_matrix(g, counts) for ai, g in _S["gt"].items() if ai.split("-", 1)[1] == stem]
    return stem, thr, k, X, mats


def pq(entries, keep_fn):
    tp_sum, tp, fp, fn = 0.0, 0, 0, 0
    for X, mats, extra in entries:
        keep = keep_fn(X, extra) if len(X) else np.zeros(0, bool)
        for m in mats:
            mm = m[:, keep] if len(X) else np.zeros((m.shape[0], 0))
            if mm.shape[1] == 0:
                fn += m.shape[0]
                continue
            hit = mm > IOU_THR
            tp_sum += mm[hit].sum()
            tp += int(hit.sum())
            fp += int((hit.sum(0) == 0).sum())
            fn += int((hit.sum(1) == 0).sum())
    den = tp + 0.5 * fp + 0.5 * fn
    return (tp_sum / den if den else 0.0), tp, fp, fn


def new_model(seed=0):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--preprocess", default="flat")
    ap.add_argument("--settings", nargs="+", default=["0.3:5", "0.4:9"], help="thr:close_k pairs")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    folds = pd.read_csv(Path(a.cache) / "folds.csv").set_index("stem")["fold"]
    stems = sorted(set().union(*[{p.stem for p in (Path(r) / "val_probs").glob("*.npy")} for r in a.runs]))
    settings = [(float(s.split(":")[0]), int(s.split(":")[1])) for s in a.settings]
    print(f"{len(stems)} images, settings {settings}")
    data = {s: {} for s in settings}
    jobs = [(st, t, k) for t, k in settings for st in stems]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.runs, a.cache, a.preprocess)) as ex:
        for stem, t, k, X, mats in ex.map(_job, jobs, chunksize=2):
            data[(t, k)][stem] = (X, mats)

    best = None
    report = []
    for (t, k), per_img in data.items():
        st_list = sorted(per_img)
        Xs, ys, gs, owner = [], [], [], []
        for st in st_list:
            X, mats = per_img[st]
            if not len(X):
                continue
            tp_any = np.mean([(m > IOU_THR).any(0) for m in mats], axis=0) if mats else np.zeros(len(X))
            Xs.append(X), ys.append(tp_any), gs.append(np.full(len(X), folds[st])), owner.append(np.full(len(X), st, dtype=object))
        X_all, y_all, g_all, own = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(owner)
        print(f"\n[thr {t}, close_k {k}] {len(X_all)} candidates, mean TP rate {y_all.mean():.3f}")

        oof = np.zeros(len(X_all))
        for f in sorted(np.unique(g_all)):
            tr, va = g_all != f, g_all == f
            oof[va] = new_model(int(f)).fit(X_all[tr], y_all[tr]).predict(X_all[va])
        score_by_img = {st: oof[own == st] for st in np.unique(own)}
        entries = [(per_img[st][0], per_img[st][1], score_by_img.get(st, np.zeros(0))) for st in st_list]

        ia, ic = FEATURES.index("log_area"), FEATURES.index("mean_prob")
        base = max(((pq(entries, lambda X, s, ma=ma, mc=mc: (np.expm1(X[:, ia]) >= ma) & (X[:, ic] >= mc)), ma, mc)
                    for ma, mc in itertools.product(MIN_AREAS, BASE_CONFS)), key=lambda r: r[0][0])
        resc = max(((pq(entries, lambda X, s, ma=ma, st_=st_: (np.expm1(X[:, ia]) >= ma) & (s >= st_)), ma, st_)
                    for ma, st_ in itertools.product(MIN_AREAS, SCORE_THRS)), key=lambda r: r[0][0])
        (bpq, btp, bfp, bfn), bma, bmc = base
        (rpq, rtp, rfp, rfn), rma, rst = resc
        print(f"  plain filter : PQ {bpq:.4f}  TP {btp} FP {bfp} FN {bfn}  (min_area {bma}, min_conf {bmc})")
        print(f"  rescorer     : PQ {rpq:.4f}  TP {rtp} FP {rfp} FN {rfn}  (min_area {rma}, score >= {rst})")
        print(f"  gain {rpq - bpq:+.4f}")
        per_fold = []
        for f in sorted(np.unique(g_all)):
            ent_f = [e for st, e in zip(st_list, entries) if folds[st] == f]
            p_b, *_ = pq(ent_f, lambda X, s: (np.expm1(X[:, ia]) >= bma) & (X[:, ic] >= bmc))
            p_r, *_ = pq(ent_f, lambda X, s: (np.expm1(X[:, ia]) >= rma) & (s >= rst))
            per_fold.append((int(f), p_b, p_r))
        print("  per fold (plain -> rescored): " + ", ".join(f"f{f} {b:.3f}->{r:.3f}" for f, b, r in per_fold))
        report.append({"thr": t, "close_k": k, "plain_pq": bpq, "rescored_pq": rpq,
                       "min_area": rma, "score_thr": float(rst)})
        if best is None or rpq > best[0]:
            best = (rpq, t, k, rma, float(rst), X_all, y_all, bpq)

    rpq, t, k, rma, rst, X_all, y_all, bpq = best
    model = new_model(0).fit(X_all, y_all)
    imp_order = None
    try:
        from sklearn.inspection import permutation_importance
        sub = np.random.default_rng(0).choice(len(X_all), min(20000, len(X_all)), replace=False)
        imp = permutation_importance(model, X_all[sub], y_all[sub], n_repeats=3, random_state=0)
        imp_order = sorted(zip(FEATURES, imp.importances_mean), key=lambda z: -z[1])
    except Exception as e:  # importance is only informative
        print("importance skipped:", e)
    bundle = {"model": model, "features": FEATURES, "thr": t, "close_k": k, "min_area": rma, "score_thr": rst}
    joblib.dump(bundle, out / "rescorer.joblib")
    with open(out / "best_post.json", "w") as f:
        json.dump({"rescorer": str(out / "rescorer.joblib"), "post": {"thr": t, "close_k": k, "min_area": rma,
                   "min_conf": 0.0}, "score_thr": rst, "val_pq": rpq, "plain_val_pq": bpq, "runs": a.runs}, f, indent=2)
    pd.DataFrame(report).to_csv(out / "rescore_report.csv", index=False)
    print(f"\nbest: thr {t}, close_k {k}, min_area {rma}, score >= {rst}: OOF PQ {rpq:.4f} (plain {bpq:.4f})")
    if imp_order:
        print("top features: " + ", ".join(f"{n} {v:.3f}" for n, v in imp_order[:8]))
    print(f"saved {out / 'rescorer.joblib'} and {out / 'best_post.json'}")


if __name__ == "__main__":
    main()
