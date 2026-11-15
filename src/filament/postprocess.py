"""Probability map -> list of disjoint filament instances.

Steps:
1. threshold the probability map (pixels that will be submitted)
2. morphological closing with an elliptical kernel, used ONLY to decide which
   nearby fragments belong to the same filament (the closed pixels are not submitted)
3. connected components of the closed map = filament ids
4. optionally grow (dilate) or shrink (erode) every instance by `grow` pixels,
   keeping instances disjoint (fixes masks that are systematically too thin or too fat)
5. drop instances that are too small or not confident enough (an uncertain
   instance costs 0.5 in the PQ denominator, often more than it can earn).
   Small instances (area < small_area) can use their own confidence threshold
   `conf_small`, because small filaments are found with lower confidence than large ones.

Area and confidence are measured on the thresholded mask, before step 4.
Instances are disjoint by construction, as Kaggle requires.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .rle import encode

_K3 = np.ones((3, 3), np.uint8)


@dataclass
class PostCfg:
    thr: float = 0.5
    close_k: int = 15
    min_area: int = 150
    min_conf: float = 0.0
    grow: int = 0                    # +n dilate / -n erode each instance by n px
    conf_small: float | None = None  # confidence threshold for instances with area < small_area
    small_area: int = 1000

    def to_dict(self):
        return asdict(self)


def _grow_labels(lab: np.ndarray, grow: int) -> np.ndarray:
    """Dilate (grow > 0) labels into background only, or erode (grow < 0) each label.

    Labels never overwrite each other, so instances stay disjoint.
    """
    if grow == 0:
        return lab
    if lab.max() >= 65535:
        raise ValueError("too many instances for uint16 morphology")
    lab = lab.astype(np.uint16)
    for _ in range(abs(grow)):
        if grow > 0:
            lab = np.where(lab == 0, cv2.dilate(lab, _K3), lab)
        else:
            # keep a pixel only if its whole 3x3 neighbourhood has the same label
            same = (cv2.erode(lab, _K3) == lab) & (cv2.dilate(lab, _K3) == lab)
            lab = np.where(same, lab, 0).astype(np.uint16)
    return lab.astype(np.int32)


def components(prob: np.ndarray, thr: float, close_k: int):
    """Threshold + grouping. Returns (binary, lab, n, stats, areas, psum) or None if empty.

    lab holds a group id on every thresholded pixel (0 elsewhere); stats are the
    bounding boxes of the closed groups.
    """
    binary = (prob > thr).astype(np.uint8)
    if not binary.any():
        return None
    if close_k > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        grouped = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ker) | binary
    else:
        grouped = binary
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    lab = np.where(binary > 0, lab, 0)
    areas = np.bincount(lab.ravel(), minlength=n)
    psum = np.bincount(lab.ravel(), weights=prob.ravel(), minlength=n)
    return binary, lab, n, stats, areas, psum


def encode_labels(lab: np.ndarray, ids, stats: np.ndarray, pad: int = 0) -> list[str]:
    """RLE counts for each label id, using its bbox (+pad) to keep encoding cheap."""
    h, w = lab.shape
    buf = np.zeros(lab.shape, np.uint8)
    out = []
    for i in ids:
        x, y, bw, bh = stats[i, :4]
        y0, y1 = max(y - pad, 0), min(y + bh + pad, h)
        x0, x1 = max(x - pad, 0), min(x + bw + pad, w)
        buf[y0:y1, x0:x1] = lab[y0:y1, x0:x1] == i
        out.append(encode(buf) if buf[y0:y1, x0:x1].any() else None)
        buf[y0:y1, x0:x1] = 0
    return out


def candidate_instances(prob: np.ndarray, thr: float, close_k: int, area_floor: int = 20,
                        grow: int = 0) -> list[dict]:
    """All instances before area/confidence filtering.

    Components smaller than `area_floor` px are dropped before RLE encoding: they can
    never clear IoU 0.5 against a real filament and only cost time (a noisy map can
    hold thousands of them).
    Each dict: counts (RLE string), area (px, before grow), conf (mean probability), bbox.
    """
    comp = components(prob, thr, close_k)
    if comp is None:
        return []
    _, lab, n, stats, areas, psum = comp

    keep_ids = [i for i in np.nonzero(areas >= max(area_floor, 1))[0] if i != 0]
    if grow:
        lab = np.where(np.isin(lab, keep_ids), lab, 0)  # dropped blobs must not block growth
        lab = _grow_labels(lab, grow)
    counts = encode_labels(lab, keep_ids, stats, pad=max(grow, 0))
    out = []
    for i, c in zip(keep_ids, counts):
        if c is None:  # erosion can remove a thin instance entirely
            continue
        x, y, bw, bh = stats[i, :4]
        out.append({"counts": c, "area": int(areas[i]), "conf": float(psum[i] / areas[i]),
                    "bbox": (int(x), int(y), int(bw), int(bh))})
    return out


def conf_thresholds(area: np.ndarray, min_conf: float, conf_small: float | None, small_area: int) -> np.ndarray:
    if conf_small is None:
        return np.full(len(area), min_conf)
    return np.where(area < small_area, conf_small, min_conf)


def filter_instances(cands: list[dict], min_area: int, min_conf: float,
                     conf_small: float | None = None, small_area: int = 1000) -> list[dict]:
    if not cands:
        return []
    area = np.array([c["area"] for c in cands])
    thr = conf_thresholds(area, min_conf, conf_small, small_area)
    return [c for c, t in zip(cands, thr) if c["area"] >= min_area and c["conf"] >= t]


def instances(prob: np.ndarray, cfg: PostCfg) -> list[dict]:
    cands = candidate_instances(prob, cfg.thr, cfg.close_k, grow=cfg.grow)
    return filter_instances(cands, cfg.min_area, cfg.min_conf, cfg.conf_small, cfg.small_area)
