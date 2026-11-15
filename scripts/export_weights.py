"""Shrink a checkpoint for the repo: fp16 weights, no optimizer state (~half the size).

  python scripts/export_weights.py runs/r34_f0/best.pt weights/r34_f0.pt
GitHub rejects files over 100 MB; check the printed size.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.model import load_checkpoint, save_checkpoint  # noqa: E402

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    model, cfg = load_checkpoint(src)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(dst, model, cfg, half=True)
    print(f"{dst}: {Path(dst).stat().st_size / 2**20:.1f} MB")
