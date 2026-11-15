"""Second-stage instance rescoring.

The segmentation network gives a probability map; post-processing turns it into
candidate instances. Instead of keeping a candidate only by its area and mean
probability, a small gradient-boosting model looks at ~20 shape / confidence /
intensity features of the candidate and predicts how likely it is to be a true
positive (IoU > 0.5 with an annotator's filament). Candidates are kept when that
score clears a threshold tuned for PQ on out-of-fold data.

Features are computed per candidate from the probability map, the preprocessed
image (uint8, 0 outside the solar disk) and the label map, all vectorised over
candidates with bincount / scipy.ndimage.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .postprocess import components, encode_labels

FEATURES = [
    "log_area", "mean_prob", "max_prob", "std_prob", "frac_p50", "frac_p70", "frac_p90",
    "bbox_w", "bbox_h", "fill", "elong", "major_len", "minor_len", "n_frag",
    "r_disk", "img_mean", "img_min", "img_std", "img_contrast", "n_cands_img",
]


def _disk_geometry(img: np.ndarray):
    disk = img > 0
    ys, xs = np.nonzero(disk[::8, ::8])
    if len(ys) == 0:
        h, w = img.shape
        return w / 2, h / 2, min(h, w) / 2
    cy, cx = ys.mean() * 8, xs.mean() * 8
    r = np.sqrt(disk.sum() / np.pi)
    return cx, cy, r


def candidate_features(prob: np.ndarray, img_u8: np.ndarray, thr: float, close_k: int,
                       area_floor: int = 20) -> tuple[list[str], np.ndarray]:
    """Candidates (RLE counts) and their feature matrix (n_cands, len(FEATURES))."""
    comp = components(prob, thr, close_k)
    if comp is None:
        return [], np.zeros((0, len(FEATURES)), np.float32)
    binary, lab, n, stats, areas, psum = comp
    ids = np.array([i for i in np.nonzero(areas >= max(area_floor, 1))[0] if i != 0], dtype=np.int64)
    if len(ids) == 0:
        return [], np.zeros((0, len(FEATURES)), np.float32)

    flat_lab = lab.ravel()
    a = areas[ids].astype(np.float64)

    def bsum(weights):
        return np.bincount(flat_lab, weights=weights.ravel(), minlength=n)[ids]

    pf = prob.astype(np.float64)
    mean_p = psum[ids] / a
    std_p = np.sqrt(np.maximum(bsum(pf * pf) / a - mean_p ** 2, 0))
    max_p = np.asarray(ndimage.maximum(prob, lab, ids))
    frac = [bsum((prob > t).astype(np.float64)) / a for t in (0.5, 0.7, 0.9)]

    # shape from second moments of the thresholded pixels of each group
    yy, xx = np.indices(lab.shape, dtype=np.float64)
    mx, my = bsum(xx) / a, bsum(yy) / a
    cxx = bsum(xx * xx) / a - mx ** 2
    cyy = bsum(yy * yy) / a - my ** 2
    cxy = bsum(xx * yy) / a - mx * my
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-6)
    major, minor = 4 * np.sqrt(l1), 4 * np.sqrt(l2)

    bw, bh = stats[ids, 2].astype(np.float64), stats[ids, 3].astype(np.float64)

    # number of separate thresholded fragments merged into each group
    frag_lab, nf = ndimage.label(binary, structure=np.ones((3, 3)))
    if nf:
        frag_group = np.zeros(nf + 1, np.int64)
        frag_group[frag_lab.ravel()] = flat_lab
        n_frag = np.bincount(frag_group[1:], minlength=n)[ids]
    else:
        n_frag = np.ones(len(ids))

    cx, cy, r = _disk_geometry(img_u8)
    r_disk = np.sqrt((mx - cx) ** 2 + (my - cy) ** 2) / max(r, 1)

    im = img_u8.astype(np.float64)
    img_mean = bsum(im) / a
    img_std = np.sqrt(np.maximum(bsum(im * im) / a - img_mean ** 2, 0))
    img_min = np.asarray(ndimage.minimum(img_u8, lab, ids), dtype=np.float64)
    disk_med = float(np.median(img_u8[img_u8 > 0][::16])) if (img_u8 > 0).any() else 128.0

    X = np.column_stack([
        np.log1p(a), mean_p, max_p, std_p, *frac,
        bw, bh, a / np.maximum(bw * bh, 1), major / np.maximum(minor, 1e-3), major, minor, n_frag,
        r_disk, img_mean, img_min, img_std, disk_med - img_mean, np.full(len(ids), len(ids)),
    ]).astype(np.float32)
    counts = encode_labels(lab, ids, stats)
    keep = [i for i, c in enumerate(counts) if c is not None]
    return [counts[i] for i in keep], X[keep]


def rescored_instances(prob: np.ndarray, img_u8: np.ndarray, bundle: dict) -> list[str]:
    """Apply a trained rescorer bundle (see scripts/train_rescorer.py). Returns RLE counts."""
    counts, X = candidate_features(prob, img_u8, bundle["thr"], bundle["close_k"])
    if not counts:
        return []
    score = bundle["model"].predict(X)
    area = np.expm1(X[:, FEATURES.index("log_area")])
    keep = (score >= bundle["score_thr"]) & (area >= bundle["min_area"])
    return [c for c, k in zip(counts, keep) if k]
