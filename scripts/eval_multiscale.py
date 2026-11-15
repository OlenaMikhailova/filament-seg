"""Multi-scale test-time augmentation on one validation fold (no training).

The network also sees each image upscaled (e.g. x1.25, x1.5): thin and small filaments then
cover more pixels. Probabilities are resized back to 2048 and averaged over the scales.
For every scale set it writes <out>/<name>/val_probs/*.npy, ready for tune_postprocess.py.

  python scripts/eval_multiscale.py --checkpoint weights/r34_f0_e80.pt --fold 0 --cache cache \
      --scales 1.0 1.25 1.5 --out runs/msx_f0

Then compare the scale sets with tune_postprocess.py on each folder
(s1 = the usual single-scale prediction, the baseline).
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.dataset import read_image  # noqa: E402
from filament.infer import predict_image  # noqa: E402
from filament.model import load_checkpoint  # noqa: E402


def predict_at_scale(model, img, scale, icfg, device):
    h, w = img.shape[:2]
    if scale == 1.0:
        return predict_image(model, img, tile=icfg["tile"], overlap=icfg["overlap"], tta=True,
                             batch=icfg["batch"], device=device)
    big = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
    p = predict_image(model, big, tile=icfg["tile"], overlap=icfg["overlap"], tta=True,
                      batch=icfg["batch"], device=device)
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.25, 1.5])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(a.checkpoint, device)
    icfg = cfg["infer"]
    img_dir = Path(a.cache) / f"img_{cfg['preprocess']}"
    folds = pd.read_csv(Path(a.cache) / "folds.csv")
    stems = folds.loc[folds["fold"] == a.fold, "stem"].tolist()

    # every subset of scales that contains 1.0, e.g. {1.0}, {1.0,1.25}, {1.0,1.5}, {1.0,1.25,1.5}
    others = [s for s in a.scales if s != 1.0]
    sets = [(1.0,) + c for r in range(len(others) + 1) for c in itertools.combinations(others, r)]
    out = Path(a.out)
    dirs = {}
    for sset in sets:
        d = out / ("s" + "_".join(f"{s:g}" for s in sset)) / "val_probs"
        d.mkdir(parents=True, exist_ok=True)
        dirs[sset] = d
    print(f"{len(stems)} images of fold {a.fold}, scale sets: {[d.parent.name for d in dirs.values()]}")

    t0 = time.time()
    for i, stem in enumerate(stems):
        img = read_image(img_dir / f"{stem}.png")
        probs = {s: predict_at_scale(model, img, s, icfg, device) for s in a.scales}
        for sset, d in dirs.items():
            p = np.mean([probs[s] for s in sset], axis=0)
            p[(img if img.ndim == 2 else img[..., 0]) == 0] = 0
            np.save(d / f"{stem}.npy", p.astype(np.float16))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(stems)} images, {time.time() - t0:.0f} s")
    print("done; now run tune_postprocess.py on each folder:")
    for d in dirs.values():
        print(f"  {d.parent}")


if __name__ == "__main__":
    main()