"""Full-resolution sliding-window inference with blended tiles and flip TTA."""
from __future__ import annotations

import numpy as np
import torch

from .dataset import disk_of, to_chw, to_input


def _starts(size: int, tile: int, stride: int) -> list[int]:
    s = list(range(0, max(size - tile, 0) + 1, stride))
    if s[-1] != size - tile:
        s.append(size - tile)
    return s


def _window(tile: int) -> np.ndarray:
    """2D Hann-like weight, never exactly zero, so tile borders count less."""
    w = np.hanning(tile + 2)[1:-1]
    return (np.outer(w, w) + 1e-3).astype(np.float32)


@torch.no_grad()
def predict_image(models, img_u8: np.ndarray, tile: int = 1024, overlap: int = 256,
                  tta: bool = True, batch: int = 4, device: str = "cuda", amp: bool = True) -> np.ndarray:
    """Average sigmoid probability over models x flips x overlapping tiles. Returns float32 (H, W)."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    x_full = to_chw(to_input(img_u8))  # (C, H, W)
    h, w = x_full.shape[1:]
    tile = min(tile, h, w)
    stride = tile - overlap
    win = _window(tile)
    coords = [(y, x) for y in _starts(h, tile, stride) for x in _starts(w, tile, stride)]
    flips = [None, (-1,), (-2,)] if tta else [None]
    use_amp = amp and device.startswith("cuda")

    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    for i in range(0, len(coords), batch):
        cb = coords[i:i + batch]
        xb = torch.from_numpy(np.stack([x_full[:, y:y + tile, x:x + tile] for y, x in cb])).to(device)
        prob = torch.zeros_like(xb[:, :1])
        n = 0
        for m in models:
            for fl in flips:
                xi = xb if fl is None else torch.flip(xb, fl)
                with torch.autocast("cuda", enabled=use_amp):
                    out = torch.sigmoid(m(xi).float())
                prob += out if fl is None else torch.flip(out, fl)
                n += 1
        prob = (prob / n)[:, 0].cpu().numpy()
        for (y, x), p in zip(cb, prob):
            acc[y:y + tile, x:x + tile] += p * win
            wsum[y:y + tile, x:x + tile] += win
    out = acc / np.maximum(wsum, 1e-6)
    out[~disk_of(img_u8)] = 0.0  # outside the solar disk
    return out
