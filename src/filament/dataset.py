"""Random-crop training dataset over cached preprocessed images and labels."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def to_input(img_u8: np.ndarray) -> np.ndarray:
    """uint8 preprocessed image (H, W) or (H, W, C) -> float32 network input (same at train and test time)."""
    return (img_u8.astype(np.float32) / 255.0 - 0.5) / 0.25


def to_chw(x: np.ndarray) -> np.ndarray:
    """(H, W) -> (1, H, W); (H, W, C) -> (C, H, W)."""
    return x[None] if x.ndim == 2 else np.transpose(x, (2, 0, 1))


def disk_of(img_u8: np.ndarray) -> np.ndarray:
    """Solar-disk mask: channel 0 > 0 (preprocessing puts 0 outside the disk)."""
    return (img_u8 if img_u8.ndim == 2 else img_u8[..., 0]) > 0


def read_image(path) -> np.ndarray:
    """Cached preprocessed image, 1 or 3 channels, exactly as written by prepare.py."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


class CropDataset(Dataset):
    """Each item is a random crop from a random training image.

    `samples_per_epoch` defines the epoch length. With probability `pos_frac` the
    crop is centred near a filament pixel, so thin structures are seen often.
    """

    def __init__(self, stems, img_dir, label_dir, crop=1024, samples_per_epoch=1600,
                 pos_frac=0.6, augment=True):
        self.stems = list(stems)
        self.img_dir, self.label_dir = Path(img_dir), Path(label_dir)
        self.crop, self.n = crop, samples_per_epoch
        self.pos_frac, self.augment = pos_frac, augment

    def __len__(self):
        return self.n

    def _load(self, stem):
        img = read_image(self.img_dir / f"{stem}.png")
        lab = cv2.imread(str(self.label_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        return img, lab

    def _crop_origin(self, lab, rng):
        h, w = lab.shape
        c = self.crop
        if rng.random() < self.pos_frac:
            ys, xs = np.nonzero(lab[::4, ::4])
            if len(ys):
                i = rng.integers(len(ys))
                cy, cx = ys[i] * 4, xs[i] * 4
                y0 = int(np.clip(cy - c // 2 + rng.integers(-c // 4, c // 4 + 1), 0, h - c))
                x0 = int(np.clip(cx - c // 2 + rng.integers(-c // 4, c // 4 + 1), 0, w - c))
                return y0, x0
        return int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        stem = self.stems[rng.integers(len(self.stems))]
        img, lab = self._load(stem)
        y0, x0 = self._crop_origin(lab, rng)
        c = self.crop
        img = img[y0:y0 + c, x0:x0 + c]
        lab = lab[y0:y0 + c, x0:x0 + c]

        if self.augment:
            k = int(rng.integers(4))
            img, lab = np.rot90(img, k), np.rot90(lab, k)
            if rng.random() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
            # contrast / brightness / noise jitter inside the disk, on the intensity channel only
            # (filter channels are derived features; they are left as computed)
            img = np.ascontiguousarray(img).copy()
            disk = disk_of(img)
            x = (img if img.ndim == 2 else img[..., 0]).astype(np.float32)
            a = rng.uniform(0.85, 1.15)
            b = rng.uniform(-10, 10)
            x = np.where(disk, np.clip((x - 128) * a + 128 + b, 1, 255), 0)
            if rng.random() < 0.3:
                x = np.where(disk, np.clip(x + rng.normal(0, rng.uniform(1, 5), x.shape), 1, 255), 0)
            if img.ndim == 2:
                img = x.astype(np.uint8)
            else:
                img[..., 0] = x.astype(np.uint8)

        x = torch.from_numpy(np.ascontiguousarray(to_chw(to_input(np.ascontiguousarray(img)))))
        y = torch.from_numpy(np.ascontiguousarray(lab).astype(np.float32) / 255.0)[None]
        return x, y