"""One-off data preparation (run once, ~10-20 min on CPU).

Creates in --cache:
  gt.csv              official-format GT (filament_id = <annotator>-<stem>_<k>) + stem column
  images.csv          one row per annotator-image
  folds.csv           one row per image file with its date-grouped fold
  img_<mode>/<stem>.png    preprocessed train images (0 outside the solar disk)
  label_<mode>/<stem>.png  training targets (soft labels stored as 0..255)

Usage:
  python scripts/prepare.py --data /path/MAGFiLO_1.0_Kaggle_2026 --cache cache --preprocess flat --labels soft union
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.data import label_for_stem, load_annotations, make_folds, paths, stems_table  # noqa: E402
from filament.preprocess import preprocess_file  # noqa: E402

_GT = None


def _init(gt_path):
    global _GT
    _GT = pd.read_csv(gt_path)


def _do_image(args):
    src, dst, mode = args
    if not Path(dst).exists():
        cv2.imwrite(dst, preprocess_file(src, mode))
    return dst


def _do_label(args):
    stem, dst, mode = args
    if not Path(dst).exists():
        lab = label_for_stem(stem, _GT, mode)
        cv2.imwrite(dst, np.round(lab * 255).astype(np.uint8))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--preprocess", nargs="+", default=["flat"])
    ap.add_argument("--labels", nargs="+", default=["soft"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    P = paths(a.data)
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)

    images_df, gt_df = load_annotations(P["train_json"])
    images_df.to_csv(cache / "images.csv", index=False)
    gt_df.to_csv(cache / "gt.csv", index=False)
    st = make_folds(stems_table(images_df), a.folds, a.seed)
    st.to_csv(cache / "folds.csv", index=False)

    print(f"annotator-images {len(images_df)} | image files {len(st)} | filaments {len(gt_df)}")
    print("annotators per image:", st["n_annotators"].value_counts().sort_index().to_dict())
    print("images per fold:", st["fold"].value_counts().sort_index().to_dict())
    areas = gt_df["segmentation_rle"].head(3000).map(
        lambda c: int(__import__("pycocotools.mask", fromlist=["area"]).area({"size": [2048, 2048], "counts": c.encode()})))
    print("filament area px (sample) percentiles 5/25/50/75/95:",
          np.percentile(areas, [5, 25, 50, 75, 95]).round().astype(int).tolist())

    with ProcessPoolExecutor(a.workers) as ex:
        for mode in a.preprocess:
            out = cache / f"img_{mode}"
            out.mkdir(exist_ok=True)
            jobs = [(str(P["train_images"] / fn), str(out / f"{Path(fn).stem}.png"), mode) for fn in st["file_name"]]
            list(ex.map(_do_image, jobs, chunksize=4))
            print(f"images [{mode}] done -> {out}")

    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(cache / "gt.csv"),)) as ex:
        for mode in a.labels:
            out = cache / f"label_{mode}"
            out.mkdir(exist_ok=True)
            jobs = [(s, str(out / f"{s}.png"), mode) for s in st["stem"]]
            list(ex.map(_do_label, jobs, chunksize=4))
            print(f"labels [{mode}] done -> {out}")


if __name__ == "__main__":
    main()
