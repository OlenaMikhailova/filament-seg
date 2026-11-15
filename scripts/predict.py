"""Predict the test set and write a validated submission CSV.

  python scripts/predict.py --checkpoints runs/r34_f0/best.pt --post runs/r34_f0/best_post.json \
      --images /path/MAGFiLO_1.0_Kaggle_2026/test/test_images --out submission.csv

Several checkpoints = probability-averaged ensemble. Prints time per image.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.infer import predict_image  # noqa: E402
from filament.model import load_checkpoint  # noqa: E402
from filament.postprocess import PostCfg, instances  # noqa: E402
from filament.preprocess import preprocess_file  # noqa: E402
from filament.rescore import rescored_instances  # noqa: E402
from filament.submission import to_dataframe, validate, write  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--post", required=True, help="best_post.json from tune_postprocess.py")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0],
                    help="multi-scale TTA, e.g. 1.0 1.5 (default: single scale)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = [load_checkpoint(p, device) for p in a.checkpoints]
    models = [m for m, _ in loaded]
    cfg = loaded[0][1]
    with open(a.post) as f:
        post_json = json.load(f)
    post = PostCfg(**post_json["post"])
    bundle = None
    if post_json.get("rescorer"):  # second-stage rescorer from train_rescorer.py
        import joblib
        rpath = Path(post_json["rescorer"])
        if not rpath.exists():
            rpath = Path(a.post).parent / rpath.name
        bundle = joblib.load(rpath)
        print(f"using rescorer {rpath} (score >= {bundle['score_thr']})")
    icfg = cfg["infer"]

    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        sys.exit(f"no .jpg/.jpeg/.png images found in {a.images!r}; nothing written")
    preds, t0 = {}, time.time()
    for p in files:
        img = preprocess_file(p, cfg["preprocess"])
        h, w = img.shape[:2]
        probs = []
        for sc in a.scales:
            x = img if sc == 1.0 else cv2.resize(img, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_LINEAR)
            pr = predict_image(models, x, tile=icfg["tile"], overlap=icfg["overlap"],
                               tta=not a.no_tta and icfg.get("tta", True), batch=icfg["batch"], device=device)
            probs.append(pr if sc == 1.0 else cv2.resize(pr, (w, h), interpolation=cv2.INTER_AREA))
        prob = np.mean(probs, axis=0)
        prob[(img if img.ndim == 2 else img[..., 0]) == 0] = 0
        if bundle is not None:
            preds[p.stem] = rescored_instances(prob, img, bundle)
        else:
            preds[p.stem] = [c["counts"] for c in instances(prob, post)]
    dt = time.time() - t0

    df = to_dataframe(preds)
    problems = validate(df)
    if df.empty:
        problems.append("no filaments predicted at all")
    if problems:
        sys.exit("submission NOT written: " + "; ".join(problems))
    write(df, a.out)
    print(f"{len(df)} filaments on {sum(bool(v) for v in preds.values())}/{len(files)} images -> {a.out}")
    print(f"inference {dt:.1f} s total, {dt / max(len(files), 1):.2f} s/image on {device}")


if __name__ == "__main__":
    main()
