"""Score a prediction CSV against a GT CSV with the official PQ, and print diagnostics.

  python scripts/evaluate_csv.py --gt cache/gt.csv --pred val_pred.csv [--stems-from cache/folds.csv --fold 0]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filament.metrics import dice_scores, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--stems-from")
    ap.add_argument("--fold", type=int)
    a = ap.parse_args()

    gt = pd.read_csv(a.gt)
    if a.stems_from is not None and a.fold is not None:
        folds = pd.read_csv(a.stems_from)
        keep = set(folds.loc[folds["fold"] == a.fold, "stem"])
        gt = gt[gt["filament_id"].str.split("_", n=1).str[0].str.split("-", n=1).str[1].isin(keep)]
    pred = pd.read_csv(a.pred)
    res = evaluate(gt[["filament_id", "segmentation_rle"]], pred)
    print(res.summary())
    d = dice_scores(gt[["filament_id", "segmentation_rle"]], pred)
    if len(d):
        print(f"Dice of overlapping pairs: mean {d.mean():.3f}, median {pd.Series(d).median():.3f}")
    print("GT 1:n degrees (0 = missed):", dict(sorted(res.gt_degree.items())))
    print("pred n:1 degrees (0 = false alarm):", dict(sorted(res.pred_degree.items())))


if __name__ == "__main__":
    main()
