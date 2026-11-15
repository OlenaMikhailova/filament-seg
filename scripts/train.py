"""Train one model.

  python scripts/train.py configs/baseline_r34.yaml
  python scripts/train.py configs/baseline_r34.yaml fold=1 work_dir=runs/r34_f1
  python scripts/train.py configs/baseline_r34.yaml train.epochs=2 train.samples_per_epoch=64   # smoke test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.train import load_config, run  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run(load_config(sys.argv[1], sys.argv[2:]))
