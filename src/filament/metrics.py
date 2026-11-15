"""Panoptic Quality exactly as in the organizers' self-evaluation notebook.

Official definition (self-evaluation notebook, Aug 2026):
- scored per annotator-image: GT ids are "<annotator>-<stem>_<k>", prediction ids "<stem>_<k>"
- a GT/prediction pair is a TP when IoU > 0.5
- FP = predictions overlapping no GT above 0.5; FN = GT with no prediction above 0.5
- PQ = sum(IoU of TP) / (|TP| + 0.5 |FP| + 0.5 |FN|), pooled over ALL annotator-images
- only annotator-images present in the GT are visited (predictions on other images are ignored)

`evaluate` computes IoU directly on RLEs with pycocotools (same pixel counts, no
2048x2048 float stacks), so it is much faster and lighter than the notebook.
`evaluate_dense` re-implements the notebook's dense path and is used only in tests.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from .rle import decode, to_coco

IOU_THR = 0.5


@dataclass
class PQResult:
    pq: float
    sq: float  # mean IoU of TPs
    rq: float  # TP / (TP + 0.5 FP + 0.5 FN)
    tp: int
    fp: int
    fn: int
    n_entries: int
    tp_ious: list = field(default_factory=list, repr=False)
    gt_degree: Counter = field(default_factory=Counter, repr=False)    # 1:n (IoU > 0)
    pred_degree: Counter = field(default_factory=Counter, repr=False)  # n:1 (IoU > 0)
    per_entry: pd.DataFrame | None = field(default=None, repr=False)

    def summary(self) -> str:
        return (f"PQ {self.pq:.4f} | SQ {self.sq:.4f} RQ {self.rq:.4f} | "
                f"TP {self.tp} FP {self.fp} FN {self.fn} | entries {self.n_entries}")


def _ids(series: pd.Series) -> pd.Series:
    return series.str.split("_", n=1).str[0]


def iou_matrix(gt_counts: list[str], pred_counts: list[str]) -> np.ndarray:
    """(n_gt, n_pred) IoU matrix from counts strings."""
    if not gt_counts or not pred_counts:
        return np.zeros((len(gt_counts), len(pred_counts)))
    gt = [to_coco(c) for c in gt_counts]
    pr = [to_coco(c) for c in pred_counts]
    # pycocotools: iou(dt, gt, iscrowd) -> (n_dt, n_gt)
    return np.asarray(mask_util.iou(pr, gt, [0] * len(gt))).T


def evaluate(gt_df: pd.DataFrame, pred_df: pd.DataFrame, keep_per_entry: bool = False) -> PQResult:
    """Official PQ (pooled over annotator-images) plus diagnostics."""
    gt_keys = _ids(gt_df["filament_id"])
    pred_keys = _ids(pred_df["filament_id"]) if len(pred_df) else pd.Series(dtype=str)
    gt_groups = gt_df.groupby(gt_keys.values)["segmentation_rle"].apply(list)
    pred_groups = pred_df.groupby(pred_keys.values)["segmentation_rle"].apply(list) if len(pred_df) else {}

    tp_ious, fp, fn = [], 0, 0
    gt_degree, pred_degree = Counter(), Counter()
    rows = []
    for ann_img, gt_counts in gt_groups.items():
        stem = ann_img.split("-", 1)[1]
        pred_counts = pred_groups.get(stem, []) if len(pred_df) else []
        n_gt, n_pred = len(gt_counts), len(pred_counts)
        e_tp, e_fp, e_fn, e_sum = 0, 0, 0, 0.0
        if n_pred == 0:
            e_fn = n_gt
            gt_degree.update([0] * n_gt)
        else:
            iou = iou_matrix(gt_counts, pred_counts)
            hit = iou > IOU_THR
            ious = iou[hit].tolist()
            tp_ious.extend(ious)
            e_tp, e_sum = len(ious), sum(ious)
            e_fp = int((hit.sum(0) == 0).sum())
            e_fn = int((hit.sum(1) == 0).sum())
            ov = iou > 0
            gt_degree.update(ov.sum(1).tolist())
            pred_degree.update(ov.sum(0).tolist())
        fp += e_fp
        fn += e_fn
        if keep_per_entry:
            den = e_tp + 0.5 * e_fp + 0.5 * e_fn
            rows.append({"annotator_image": ann_img, "stem": stem, "n_gt": n_gt, "n_pred": n_pred,
                         "tp": e_tp, "fp": e_fp, "fn": e_fn, "pq": e_sum / den if den else 0.0})

    tp = len(tp_ious)
    den = tp + 0.5 * fp + 0.5 * fn
    pq = sum(tp_ious) / den if den > 0 else 0.0
    sq = float(np.mean(tp_ious)) if tp else 0.0
    rq = tp / den if den > 0 else 0.0
    return PQResult(pq, sq, rq, tp, fp, fn, len(gt_groups), tp_ious, gt_degree, pred_degree,
                    pd.DataFrame(rows) if keep_per_entry else None)


def evaluate_dense(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """Dense re-implementation of the notebook (decode masks, matmul). Slow; tests only."""
    gt_keys, pred_keys = _ids(gt_df["filament_id"]), _ids(pred_df["filament_id"])
    tp_ious, fp, fn = [], 0, 0
    for ann_img in gt_keys.unique():
        stem = ann_img.split("-", 1)[1]
        g = np.stack([decode(c).ravel() for c in gt_df.loc[gt_keys == ann_img, "segmentation_rle"]]).astype(np.float64)
        p_list = pred_df.loc[pred_keys == stem, "segmentation_rle"].tolist()
        if not p_list:
            fn += len(g)
            continue
        p = np.stack([decode(c).ravel() for c in p_list]).astype(np.float64)
        inter = g @ p.T
        union = g.sum(1)[:, None] + p.sum(1)[None, :] - inter
        iou = np.where(union == 0, 0, inter / np.maximum(union, 1))
        hit = iou > IOU_THR
        tp_ious.extend(iou[hit].tolist())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    den = len(tp_ious) + 0.5 * fp + 0.5 * fn
    return sum(tp_ious) / den if den > 0 else 0.0


def dice_scores(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
    """Dice of every overlapping GT/prediction pair (what the rubric's Dice histogram shows)."""
    gt_keys, pred_keys = _ids(gt_df["filament_id"]), _ids(pred_df["filament_id"])
    out = []
    for ann_img in gt_keys.unique():
        stem = ann_img.split("-", 1)[1]
        gc = gt_df.loc[gt_keys == ann_img, "segmentation_rle"].tolist()
        pc = pred_df.loc[pred_keys == stem, "segmentation_rle"].tolist()
        iou = iou_matrix(gc, pc)
        iou = iou[iou > 0]
        out.extend((2 * iou / (1 + iou)).tolist())  # Dice = 2 IoU / (1 + IoU)
    return np.asarray(out)
