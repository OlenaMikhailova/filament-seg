"""Loading MAGFiLO annotations and building leak-free folds.

Key facts from the competition Data tab:
- image["id"] = "<annotator>-<image stem>", e.g. "010401-20160920230134Lh"
- one image file can be annotated by up to 3 annotators independently
- the metric scores every annotator-image separately, so the GT table keeps them apart
- each annotation has exactly one polygon
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import IMG_SIZE
from .rle import decode, polygon_to_counts

TRAIN_JSON = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def paths(data_root: str | Path) -> dict[str, Path]:
    root = Path(data_root)
    return {
        "train_images": root / "train" / "train_images",
        "train_json": root / "train" / TRAIN_JSON,
        "test_images": root / "test" / "test_images",
    }


def load_annotations(json_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the COCO JSON.

    Returns
    -------
    images_df : one row per annotator-image
        columns: annotator_image, stem, file_name, date, year
    gt_df : one row per annotated filament, in the official evaluation format
        columns: filament_id ("<annotator_image>_<k>"), segmentation_rle, annotator_image, stem
    """
    with open(json_path) as f:
        coco = json.load(f)

    img_rows = []
    by_id = {}
    for im in coco["images"]:
        ann_img = str(im["id"])
        stem = Path(im["file_name"]).stem
        # sanity: id should be "<annotator>-<stem>"
        if "-" not in ann_img or ann_img.split("-", 1)[1] != stem:
            ann_img = f"{ann_img.split('-', 1)[0]}-{stem}"
        h, w = im.get("height", IMG_SIZE), im.get("width", IMG_SIZE)
        row = {"annotator_image": ann_img, "stem": stem, "file_name": im["file_name"],
               "date": stem[:8], "year": int(stem[:4]), "height": h, "width": w}
        img_rows.append(row)
        by_id[str(im["id"])] = row
    images_df = pd.DataFrame(img_rows).drop_duplicates("annotator_image").reset_index(drop=True)

    counters: dict[str, int] = {}
    gt_rows = []
    for ann in coco["annotations"]:
        im = by_id[str(ann["image_id"])]
        seg = ann["segmentation"]
        polygon = seg[0] if isinstance(seg[0], list) else seg
        if len(polygon) < 6:  # fewer than 3 points
            continue
        k = counters.get(im["annotator_image"], 0) + 1
        counters[im["annotator_image"]] = k
        gt_rows.append({
            "filament_id": f"{im['annotator_image']}_{k}",
            "segmentation_rle": polygon_to_counts(polygon, im["height"], im["width"]),
            "annotator_image": im["annotator_image"],
            "stem": im["stem"],
        })
    gt_df = pd.DataFrame(gt_rows)
    return images_df, gt_df


def stems_table(images_df: pd.DataFrame) -> pd.DataFrame:
    """One row per image file (stem) with its number of annotators."""
    t = (images_df.groupby("stem")
         .agg(date=("date", "first"), year=("year", "first"), file_name=("file_name", "first"),
              n_annotators=("annotator_image", "nunique"))
         .reset_index())
    return t


def make_folds(stems_df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Assign a fold to each image stem.

    Grouped by observation date (frames from the same day are near duplicates and
    would leak), stratified by year so every fold covers the solar cycle.
    """
    df = stems_df.copy()
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(df, y=df["year"], groups=df["date"])):
        df.loc[df.index[val_idx], "fold"] = fold
    return df


def label_for_stem(stem: str, gt_df: pd.DataFrame, mode: str = "soft",
                   h: int = IMG_SIZE, w: int = IMG_SIZE) -> np.ndarray:
    """Semantic training target for one image file, combining all its annotators.

    mode:
      "soft"     fraction of annotators marking each pixel (0..1)
      "union"    any annotator
      "majority" at least half of annotators
    Returns float32 (H, W).
    """
    sub = gt_df[gt_df["stem"] == stem]
    ann_imgs = sub["annotator_image"].unique()
    if len(ann_imgs) == 0:
        return np.zeros((h, w), np.float32)
    acc = np.zeros((h, w), np.float32)
    for ai in ann_imgs:
        m = np.zeros((h, w), bool)
        for counts in sub.loc[sub["annotator_image"] == ai, "segmentation_rle"]:
            m |= decode(counts, h, w).astype(bool)
        acc += m
    frac = acc / len(ann_imgs)
    if mode == "soft":
        return frac
    if mode == "union":
        return (frac > 0).astype(np.float32)
    if mode == "majority":
        return (frac >= 0.5).astype(np.float32)
    raise ValueError(f"unknown label mode {mode}")
