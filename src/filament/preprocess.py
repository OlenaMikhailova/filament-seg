"""Image preprocessing: solar-disk masking and limb-darkening flattening.

Output of `preprocess` is a uint8 image where pixels outside the solar disk are 0 and
pixels inside are in [1, 255]. That way the disk mask is recoverable as `img > 0`
and the same function serves training (cached to PNG) and test-time inference.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def solar_disk(img: np.ndarray, shrink: float = 0.995) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Find the solar disk; returns (bool mask, (cx, cy, r)) in full-resolution pixels."""
    h, w = img.shape
    scale = 512 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        cx, cy, r = w / 2, h / 2, min(h, w) / 2
    else:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cnts, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
        cx, cy, r = cx / scale, cy / scale, r / scale
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * shrink) ** 2
    return mask, (float(cx), float(cy), float(r))


def flatten(img: np.ndarray, disk: np.ndarray, sigma_full: float = 96.0) -> np.ndarray:
    """Divide by a smooth background (removes limb darkening and large-scale gradients).

    The background is a normalised (masked) Gaussian blur computed at 1/8 resolution
    for speed. Returns a float32 contrast image, ~1.0 on quiet Sun, < 1 on filaments.
    """
    h, w = img.shape
    f = img.astype(np.float32)
    s = 8
    small = cv2.resize(f, (w // s, h // s), interpolation=cv2.INTER_AREA)
    dsm = cv2.resize(disk.astype(np.float32), (w // s, h // s), interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(small * dsm, (0, 0), sigma_full / s)
    den = cv2.GaussianBlur(dsm, (0, 0), sigma_full / s)
    bg = cv2.resize(num / np.maximum(den, 1e-3), (w, h), interpolation=cv2.INTER_LINEAR)
    out = f / np.maximum(bg, 1.0)
    out[~disk] = 1.0
    return out


def _to_u8(resp: np.ndarray, disk: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """Non-negative filter response -> uint8, scaled by a robust percentile inside the disk."""
    hi = np.percentile(resp[disk][::7], pct) + 1e-6
    out = np.clip(resp / hi * 255.0, 0, 255).astype(np.uint8)
    out[~disk] = 0
    return out


def dark_ridges(img_f: np.ndarray, sigmas=(2.0, 4.0, 8.0)) -> np.ndarray:
    """Multi-scale Hessian ridge filter for thin DARK lines (Sato-style, as for vessels).

    At each scale the largest Hessian eigenvalue is positive across a dark line;
    its scale-normalised value (sigma^2 * lambda_max, only where positive) is taken,
    and the maximum over scales is returned.
    """
    out = np.zeros_like(img_f, dtype=np.float32)
    for s in sigmas:
        g = cv2.GaussianBlur(img_f, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        tr_half = (dxx + dyy) / 2
        lam_max = tr_half + np.sqrt(((dxx - dyy) / 2) ** 2 + dxy ** 2)
        np.maximum(out, (s * s) * np.maximum(lam_max, 0), out=out)
    return out


def black_tophat(img_u8: np.ndarray, size: int = 51, smooth: float = 2.0) -> np.ndarray:
    """Dark structures narrower than `size` px, relative to their local background.

    The image is lightly smoothed first so pixel noise does not dominate the response.
    """
    g = cv2.GaussianBlur(img_u8, (0, 0), smooth) if smooth > 0 else img_u8
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, ker).astype(np.float32)


def preprocess(img: np.ndarray, mode: str = "flat") -> np.ndarray:
    """Raw grayscale uint8 -> model-ready uint8 (0 outside disk, 1..255 inside).

    mode "flat" : flattened contrast, z-scored inside the disk, mapped 128 + 32 z   -> (H, W)
    mode "raw"  : original intensities inside the disk                                -> (H, W)
    mode "ridge": (H, W, 3) = [flat, dark-ridge filter, black top-hat]; channel 0 is
                  the "flat" image, so the disk mask is still channel 0 > 0
    """
    disk, _ = solar_disk(img)
    if mode == "raw":
        out = np.clip(img, 1, 255).astype(np.uint8)
    elif mode in ("flat", "ridge"):
        c = flatten(img, disk)
        vals = c[disk]
        mu, sd = np.median(vals), vals.std() + 1e-6
        z = (c - mu) / sd
        out = np.clip(128 + 32 * z, 1, 255).astype(np.uint8)
    else:
        raise ValueError(f"unknown preprocess mode {mode}")
    out[~disk] = 0
    if mode == "ridge":
        inner = cv2.erode(disk.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0  # no limb edge response
        filled = np.where(disk, out, 128).astype(np.uint8)  # avoid a fake dark ring at the limb
        ridge = _to_u8(dark_ridges(filled.astype(np.float32)), inner)
        that = _to_u8(black_tophat(filled), inner)
        out = np.dstack([out, ridge, that])
    return out


def preprocess_file(path: str | Path, mode: str = "flat") -> np.ndarray:
    return preprocess(read_gray(path), mode)