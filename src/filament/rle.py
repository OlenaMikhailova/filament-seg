"""Binary mask <-> COCO compressed RLE "counts" string, as used in the submission CSV.

The competition stores only the `counts` field; the size is always 2048 x 2048.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

from . import IMG_SIZE


def encode(mask: np.ndarray) -> str:
    """(H, W) bool/uint8 mask -> RLE counts string (ascii, no quotes)."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def to_coco(counts: str, h: int = IMG_SIZE, w: int = IMG_SIZE) -> dict:
    """Counts string -> pycocotools RLE object (for iou/area without decoding)."""
    return {"size": [h, w], "counts": counts.encode("ascii")}


def decode(counts: str, h: int = IMG_SIZE, w: int = IMG_SIZE) -> np.ndarray:
    """Counts string -> (H, W) uint8 mask."""
    return mask_util.decode(to_coco(counts, h, w))


def polygon_to_counts(polygon: list[float], h: int = IMG_SIZE, w: int = IMG_SIZE) -> str:
    """COCO polygon [x0, y0, x1, y1, ...] -> counts string (same path pycocotools annToMask uses)."""
    rles = mask_util.frPyObjects([polygon], h, w)
    rle = mask_util.merge(rles)
    return rle["counts"].decode("ascii")
